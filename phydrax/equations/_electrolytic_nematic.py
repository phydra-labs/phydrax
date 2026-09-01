#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._electrochemistry import (
    AbstractElectrochemicalClosure,
    ElectrolyteTransportParameters,
)
from ._nematic import (
    LandauDeGennesClosure,
    LandauDeGennesParameters,
    NematicThermodynamicFields,
)


class ElectrolyticNematicParameters(StrictModule, NonTrainableState):
    nematic: LandauDeGennesParameters
    electrolyte: ElectrolyteTransportParameters
    ion_nematic_coupling: Array
    isotropic_permittivity: Array
    anisotropic_permittivity: Array
    parameters_id: str = eqx.field(static=True)

    def __init__(
        self,
        nematic: LandauDeGennesParameters,
        electrolyte: ElectrolyteTransportParameters,
        ion_nematic_coupling: ArrayLike,
        isotropic_permittivity: ArrayLike,
        anisotropic_permittivity: ArrayLike,
        /,
    ):
        if not isinstance(nematic, LandauDeGennesParameters):
            raise TypeError("nematic must be LandauDeGennesParameters.")
        if not isinstance(electrolyte, ElectrolyteTransportParameters):
            raise TypeError("electrolyte must be ElectrolyteTransportParameters.")
        coupling = jnp.asarray(ion_nematic_coupling)
        isotropic = jnp.asarray(isotropic_permittivity, dtype=coupling.dtype)
        anisotropic = jnp.asarray(anisotropic_permittivity, dtype=coupling.dtype)
        if coupling.shape != (electrolyte.schema.species_count,) or (
            isotropic.shape != () or anisotropic.shape != ()
        ):
            raise ValueError("Electrolytic nematic parameter shapes are invalid.")
        if not bool(
            jnp.all(jnp.isfinite(coupling))
            & jnp.isfinite(isotropic)
            & (isotropic > 0.0)
            & jnp.isfinite(anisotropic)
        ):
            raise ValueError("Electrolytic nematic parameters are inadmissible.")
        self.nematic = nematic
        self.electrolyte = electrolyte
        self.ion_nematic_coupling = coupling
        self.isotropic_permittivity = isotropic
        self.anisotropic_permittivity = anisotropic
        self.parameters_id = canonical_fingerprint(
            {
                "kind": "electrolytic-nematic-parameters",
                "electrolyte": electrolyte.parameters_id,
                "coupling": array_tree_fingerprint(np.asarray(coupling)),
                "permittivity": [float(isotropic), float(anisotropic)],
            }
        )


class ElectrolyticNematicFields(StrictModule):
    nematic: NematicThermodynamicFields
    total_free_energy_density: Array
    molecular_field: Array
    ionic_electrochemical_potential: Array
    charge_density: Array
    permittivity_tensor: Array
    total_stress: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


class ElectrolyticNematicClosure(StrictModule, NonTrainableState):
    nematic: LandauDeGennesClosure
    electrochemical: AbstractElectrochemicalClosure
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        nematic: LandauDeGennesClosure,
        electrochemical: AbstractElectrochemicalClosure,
        /,
    ):
        if not isinstance(nematic, LandauDeGennesClosure):
            raise TypeError("nematic must be LandauDeGennesClosure.")
        if not isinstance(electrochemical, AbstractElectrochemicalClosure):
            raise TypeError(
                "electrochemical must implement AbstractElectrochemicalClosure."
            )
        self.nematic = nematic
        self.electrochemical = electrochemical
        self.closure_id = canonical_fingerprint(
            {
                "kind": "electrolytic-nematic-closure",
                "nematic": nematic.closure_id,
                "electrochemical": electrochemical.closure_id,
            }
        )

    def evaluate(
        self,
        compact_q: ArrayLike,
        compact_gradient: ArrayLike,
        compact_laplacian: ArrayLike,
        concentrations: ArrayLike,
        potential: ArrayLike,
        electric_field: ArrayLike,
        parameters: ElectrolyticNematicParameters,
        /,
        *,
        fixed_charge: ArrayLike = 0.0,
    ) -> ElectrolyticNematicFields:
        concentration = jnp.asarray(concentrations)
        electric = jnp.asarray(electric_field, dtype=concentration.dtype)
        nematic = self.nematic.evaluate(
            compact_q,
            compact_gradient,
            compact_laplacian,
            parameters.nematic,
            electric_field=electric,
        )
        electrochemical = self.electrochemical.evaluate(
            concentration,
            potential,
            parameters.electrolyte,
            fixed_charge=fixed_charge,
        )
        trace_q2 = jnp.sum(nematic.tensor * nematic.tensor, axis=(-2, -1))
        coupling_density = contract(
            "...s,s->...", concentration, parameters.ion_nematic_coupling
        )
        coupling_energy = coupling_density * trace_q2
        coupling_molecular_tensor = (
            -2.0 * coupling_density[..., None, None] * nematic.tensor
        )
        coupling_molecular = self.nematic.basis.encode(coupling_molecular_tensor)
        molecular = nematic.molecular_field + coupling_molecular
        ionic = electrochemical.electrochemical_potential + (
            trace_q2[..., None] * parameters.ion_nematic_coupling
        )
        dimension = self.nematic.basis.orientation_dimension
        identity = jnp.eye(dimension, dtype=concentration.dtype)
        permittivity = (
            parameters.isotropic_permittivity * identity
            + parameters.anisotropic_permittivity * nematic.tensor
        )
        electric_displacement = contract("...ij,...j->...i", permittivity, electric)
        electric_energy = 0.5 * jnp.sum(electric * electric_displacement, axis=-1)
        maxwell = (
            contract("...i,...j->...ij", electric_displacement, electric)
            - electric_energy[..., None, None] * identity
        )
        osmotic = -electrochemical.osmotic_pressure[..., None, None] * identity
        total_energy = (
            nematic.total_energy_density
            + electrochemical.chemical_free_energy_density
            + coupling_energy
            + electric_energy
        )
        total_stress = (
            nematic.distortion_stress + nematic.electric_stress + maxwell + osmotic
        )
        tensor_norm = jnp.sqrt(jnp.sum(nematic.tensor * nematic.tensor, axis=(-2, -1)))
        permittivity_margin = (
            parameters.isotropic_permittivity
            - jnp.abs(parameters.anisotropic_permittivity) * tensor_norm
        )
        successful = (
            nematic.successful
            & electrochemical.successful
            & jnp.all(jnp.isfinite(total_energy))
            & jnp.all(jnp.isfinite(molecular))
            & jnp.all(jnp.isfinite(total_stress))
            & jnp.all(permittivity_margin > 0.0)
        )
        return ElectrolyticNematicFields(
            nematic,
            total_energy,
            molecular,
            ionic,
            electrochemical.charge_density,
            permittivity,
            total_stress,
            successful,
            self.closure_id,
        )


__all__ = [
    "ElectrolyticNematicClosure",
    "ElectrolyticNematicFields",
    "ElectrolyticNematicParameters",
]

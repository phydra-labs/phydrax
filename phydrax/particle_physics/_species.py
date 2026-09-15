#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import CHARGE, ENERGY, UnitDefinition
from ._identity import ParticleCatalogueReference


class ParticleSpeciesTable(StrictModule, NonTrainableState):
    """Bounded rest-energy and charge lookup keyed by immutable PDG identities."""

    pdg_ids: Array
    rest_energies: Array
    charges: Array
    active: Array
    catalogue: ParticleCatalogueReference
    energy_unit: UnitDefinition
    charge_unit: UnitDefinition
    capacity: int = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        pdg_ids: ArrayLike,
        rest_energies: ArrayLike,
        charges: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
        catalogue: ParticleCatalogueReference,
        energy_unit: UnitDefinition,
        charge_unit: UnitDefinition,
    ):
        identifiers = np.asarray(pdg_ids)
        energies = np.asarray(rest_energies)
        charges_ = np.asarray(charges)
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError("pdg_ids must be a non-empty one-dimensional array.")
        if not np.issubdtype(identifiers.dtype, np.integer):
            raise TypeError("pdg_ids must contain integers.")
        if energies.shape != identifiers.shape or charges_.shape != identifiers.shape:
            raise ValueError("Species energies and charges must align with pdg_ids.")
        if not isinstance(catalogue, ParticleCatalogueReference):
            raise TypeError("catalogue must be ParticleCatalogueReference.")
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise ValueError("energy_unit must have energy dimension.")
        if not isinstance(charge_unit, UnitDefinition) or charge_unit.dimension != CHARGE:
            raise ValueError("charge_unit must have charge dimension.")
        active_ = (
            np.ones(identifiers.shape, dtype=bool)
            if active is None
            else np.asarray(active, dtype=bool)
        )
        if active_.shape != identifiers.shape:
            raise ValueError("active must align with pdg_ids.")
        if np.any(~np.isfinite(energies[active_])) or np.any(energies[active_] < 0.0):
            raise ValueError("Active rest energies must be finite and nonnegative.")
        if np.any(~np.isfinite(charges_[active_])):
            raise ValueError("Active charges must be finite.")
        if len(set(identifiers[active_].tolist())) != int(np.sum(active_)):
            raise ValueError("Active PDG identities must be unique.")
        self.pdg_ids = jnp.asarray(identifiers, dtype=jnp.int32)
        self.rest_energies = jnp.asarray(energies)
        self.charges = jnp.asarray(charges_, dtype=self.rest_energies.dtype)
        self.active = jnp.asarray(active_)
        self.catalogue = catalogue
        self.energy_unit = energy_unit
        self.charge_unit = charge_unit
        self.capacity = int(identifiers.size)
        self.table_id = canonical_fingerprint(
            {
                "kind": "particle-species-table",
                "catalogue": catalogue.catalogue_id,
                "energy_unit": energy_unit.unit_id,
                "charge_unit": charge_unit.unit_id,
                "content": array_tree_fingerprint(
                    {
                        "pdg_ids": identifiers,
                        "rest_energies": energies,
                        "charges": charges_,
                        "active": active_,
                    }
                ),
            }
        )


class ParticleSpeciesLookup(StrictModule, NonTrainableState):
    rest_energies: Array
    charges: Array
    found: Array
    table_id: str = eqx.field(static=True)


def lookup_particle_species(
    table: ParticleSpeciesTable, pdg_ids: ArrayLike, /
) -> ParticleSpeciesLookup:
    """Lookup arbitrary-shaped integer identities without host callbacks."""
    if not isinstance(table, ParticleSpeciesTable):
        raise TypeError("table must be ParticleSpeciesTable.")
    requested = jnp.asarray(pdg_ids)
    if not jnp.issubdtype(requested.dtype, jnp.integer):
        raise TypeError("pdg_ids must contain integers.")
    matches = (requested[..., None] == table.pdg_ids) & table.active
    found = jnp.any(matches, axis=-1)
    indices = jnp.argmax(matches, axis=-1)
    energies = jnp.where(found, table.rest_energies[indices], jnp.nan)
    charges = jnp.where(found, table.charges[indices], jnp.nan)
    return ParticleSpeciesLookup(energies, charges, found, table.table_id)


__all__ = [
    "ParticleSpeciesLookup",
    "ParticleSpeciesTable",
    "lookup_particle_species",
]

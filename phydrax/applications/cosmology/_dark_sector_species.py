#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class DarkSectorSpeciesPlan(StrictModule, NonTrainableState):
    """Immutable microscopic dark-sector species and conserved-charge identity.

    ``mass`` is a microscopic inertial/rest mass, not a simulation-packet mass.
    ``internal_energy`` uses the declared physical-energy unit and is measured
    relative to the species family's chosen ground-state zero.  Charge columns
    are named explicitly so reaction plans can compare like conserved quantities.
    """

    species_id: str = eqx.field(static=True)
    mass: float = eqx.field(static=True)
    internal_energy: float = eqx.field(static=True)
    degeneracy: int = eqx.field(static=True)
    charge_names: tuple[str, ...] = eqx.field(static=True)
    charges: Array
    mass_unit: str = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    species_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_id: str,
        mass: float,
        /,
        *,
        internal_energy: float = 0.0,
        degeneracy: int = 1,
        charge_names: Sequence[str] = (),
        charges: ArrayLike = (),
        mass_unit: str = "physical-mass",
        energy_unit: str = "physical-energy",
    ):
        identifier = str(species_id).strip()
        mass_ = float(mass)
        energy = float(internal_energy)
        if isinstance(degeneracy, bool) or not isinstance(degeneracy, Integral):
            raise TypeError("Dark-sector degeneracy must be an integer.")
        degeneracy_ = int(degeneracy)
        names = tuple(str(name).strip() for name in charge_names)
        charge_values = np.asarray(charges, dtype=np.float64)
        mass_unit_ = str(mass_unit).strip()
        energy_unit_ = str(energy_unit).strip()
        if not identifier:
            raise ValueError("Dark-sector species_id must be non-empty.")
        if not np.isfinite(mass_) or mass_ <= 0.0:
            raise ValueError("Dark-sector microscopic mass must be finite and positive.")
        if not np.isfinite(energy):
            raise ValueError("Dark-sector internal energy must be finite.")
        if degeneracy_ < 1:
            raise ValueError("Dark-sector degeneracy must be positive.")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Dark-sector charge names must be non-empty and unique.")
        if charge_values.shape != (len(names),):
            raise ValueError("Dark-sector charges must have one value per charge name.")
        if not np.all(np.isfinite(charge_values)):
            raise ValueError("Dark-sector charges must be finite.")
        if not mass_unit_ or not energy_unit_:
            raise ValueError("Dark-sector mass and energy units must be explicit.")

        charge_array = jax.lax.stop_gradient(jnp.asarray(charge_values))
        self.species_id = identifier
        self.mass = mass_
        self.internal_energy = energy
        self.degeneracy = degeneracy_
        self.charge_names = names
        self.charges = charge_array
        self.mass_unit = mass_unit_
        self.energy_unit = energy_unit_
        self.species_plan_id = canonical_fingerprint(
            {
                "kind": "dark-sector-microscopic-species",
                "species_id": identifier,
                "mass": mass_,
                "internal_energy": energy,
                "degeneracy": degeneracy_,
                "charge_names": list(names),
                "charges": array_tree_fingerprint(charge_array),
                "mass_unit": mass_unit_,
                "energy_unit": energy_unit_,
            }
        )

    def charge(self, charge_name: str, /) -> Array:
        """Return one named conserved charge without runtime string dispatch."""

        name = str(charge_name).strip()
        if name not in self.charge_names:
            raise ValueError(f"Unknown dark-sector charge {name!r}.")
        return self.charges[self.charge_names.index(name)]


__all__ = ["DarkSectorSpeciesPlan"]

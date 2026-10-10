#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declared ternary-fission and decay helium production on native inventories.

Yield data are supplied with evaluated-data provenance; no undocumented
universal ternary yield is bundled. Decay production is obtained from existing
ledger-validated InventoryTransition products, retaining nuclide/network
identity rather than treating every decay as an alpha event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..sparse import EdgeRelation, SparseLinearMap
from ._activation import AVOGADRO_PER_MOL, InventoryTransition, NuclideInventory
from ._fission_products import FISSIONING_NUCLIDES, FissioningNuclide
from ._identity import NuclideKey
from ._provenance import NuclearDataProvenance


@final
class HeliumProductionRates(StrictModule):
    """Helium atoms produced per reference volume and time [atoms/m³/s]."""

    ternary: Array
    decay: Array

    @property
    def total(self) -> Array:
        return self.ternary + self.decay


@final
class HeliumProductionPlan(StrictModule):
    """Helium source bound to fissioning identities and a decay inventory.

    ``ternary_yields`` are atoms of the declared helium isotope per fission,
    one value for every canonical fissioning nuclide. ``decay_transitions``
    must be decay-only transitions from the native inventory owner. The
    helium isotope, inventory volume, and nuclear-data provenance are explicit.
    """

    ternary_yields: Array
    decay_action: SparseLinearMap
    helium: NuclideKey = eqx.field(static=True)
    inventory_nuclide_ids: tuple[str, ...] = eqx.field(static=True)
    inventory_network_id: str = eqx.field(static=True)
    provenance: NuclearDataProvenance = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ternary_yields: Mapping[FissioningNuclide, float],
        inventory_template: NuclideInventory,
        decay_transitions: Sequence[InventoryTransition],
        provenance: NuclearDataProvenance,
        /,
        *,
        helium: NuclideKey,
    ) -> None:
        if not isinstance(inventory_template, NuclideInventory):
            raise TypeError("inventory_template must be NuclideInventory.")
        if not isinstance(provenance, NuclearDataProvenance):
            raise TypeError("provenance must be NuclearDataProvenance.")
        if not isinstance(helium, NuclideKey):
            raise TypeError("helium must be NuclideKey.")
        if helium.proton_number != 2:
            raise ValueError("helium must identify an explicit helium nuclide.")
        if not isinstance(ternary_yields, Mapping):
            raise TypeError("ternary_yields must be a nuclide-keyed mapping.")
        if set(ternary_yields) != set(FISSIONING_NUCLIDES):
            raise ValueError(
                "Ternary yields must declare every fissioning nuclide exactly."
            )
        yields = np.asarray(
            [ternary_yields[name] for name in FISSIONING_NUCLIDES], dtype=np.float64
        )
        if np.any(~np.isfinite(yields)) or np.any(yields < 0.0):
            raise ValueError("Ternary helium yields must be finite and nonnegative.")
        transitions = tuple(decay_transitions)
        if any(not isinstance(item, InventoryTransition) for item in transitions):
            raise TypeError("decay_transitions must contain InventoryTransition values.")
        if len({item.transition_id for item in transitions}) != len(transitions):
            raise ValueError("Helium decay transitions must be unique.")
        transitions = tuple(sorted(transitions, key=lambda item: item.transition_id))
        indices: list[int] = []
        rates: list[float] = []
        for transition in transitions:
            if transition.flux_driven:
                raise ValueError(
                    "The decay helium source cannot contain flux-driven reactions."
                )
            parent = transition.parent.nuclide_id
            if parent not in inventory_template.nuclide_ids:
                raise ValueError("A helium-source parent is absent from the inventory.")
            products = dict(transition.products)
            if helium not in products or products[helium] <= 0.0:
                raise ValueError(
                    "A helium-source transition must produce the declared isotope."
                )
            indices.append(inventory_template.nuclide_ids.index(parent))
            rates.append(transition.decay_rate_s * products[helium] * AVOGADRO_PER_MOL)
        relation = EdgeRelation(
            np.asarray(indices, dtype=np.int32),
            np.zeros((len(indices),), dtype=np.int32),
            source_size=len(inventory_template.nuclide_ids),
            target_size=1,
        )
        action = SparseLinearMap(relation, np.asarray(rates, dtype=np.float64))
        self.ternary_yields = jnp.asarray(yields)
        self.decay_action = action
        self.helium = helium
        self.inventory_nuclide_ids = inventory_template.nuclide_ids
        self.inventory_network_id = inventory_template.network_id
        self.provenance = provenance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nuclear-helium-production",
                "helium": helium.nuclide_id,
                "fissioning_nuclides": FISSIONING_NUCLIDES,
                "ternary_yields": yields,
                "inventory_network": inventory_template.network_id,
                "inventory_nuclides": inventory_template.nuclide_ids,
                "transitions": [item.transition_id for item in transitions],
                "provenance": provenance.provenance_id,
            }
        )

    def evaluate(
        self,
        fission_rate_density: ArrayLike,
        fission_fractions: ArrayLike,
        inventory: NuclideInventory,
        reference_volume: ArrayLike,
        /,
    ) -> HeliumProductionRates:
        """Source rates of the declared isotope; decay inventory is in moles."""

        if not isinstance(inventory, NuclideInventory):
            raise TypeError("inventory must be NuclideInventory.")
        if (
            inventory.network_id != self.inventory_network_id
            or inventory.nuclide_ids != self.inventory_nuclide_ids
        ):
            raise ValueError("Helium production inventory identity changed.")
        fractions = jnp.asarray(fission_fractions)
        if fractions.shape[-1:] != (len(FISSIONING_NUCLIDES),):
            raise ValueError(
                "Fission fractions require the canonical fissioning-nuclide axis."
            )
        fractions = eqx.error_if(
            fractions,
            jnp.any(~jnp.isfinite(fractions) | (fractions < 0.0))
            | jnp.any(jnp.abs(jnp.sum(fractions, axis=-1) - 1.0) > 1.0e-12),
            "Fission fractions must be finite, nonnegative, and normalized.",
        )
        rate = jnp.asarray(fission_rate_density, dtype=fractions.dtype)
        rate = eqx.error_if(
            rate,
            jnp.any(~jnp.isfinite(rate) | (rate < 0.0)),
            "Fission-rate density must be finite and nonnegative.",
        )
        volume = jnp.asarray(reference_volume, dtype=fractions.dtype)
        if volume.shape != ():
            raise ValueError(
                "The declared decay inventory requires one reference volume."
            )
        volume = eqx.error_if(
            volume,
            ~jnp.isfinite(volume) | (volume <= 0.0),
            "Reference inventory volume must be finite and positive.",
        )
        amounts = eqx.error_if(
            inventory.amounts_mol,
            jnp.any(~jnp.isfinite(inventory.amounts_mol) | (inventory.amounts_mol < 0.0)),
            "Decay inventory amounts must be finite and nonnegative.",
        )
        ternary = rate * (fractions @ self.ternary_yields)
        decay = self.decay_action.mv(amounts)[0] / volume
        return HeliumProductionRates(ternary, decay)


__all__ = ["HeliumProductionPlan", "HeliumProductionRates"]

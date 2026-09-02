#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ChemicalComponentCatalog(StrictModule, NonTrainableState):
    """Canonical chemical identities shared by phase-specific species occurrences."""

    component_names: tuple[str, ...] = eqx.field(static=True)
    molar_masses: Float[Array, " component"]
    element_names: tuple[str, ...] = eqx.field(static=True)
    element_composition: Int[Array, "element component"]
    charges: Int[Array, " component"]
    provenance: str = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    element_count: int = eqx.field(static=True)
    catalog_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_names: Sequence[str],
        molar_masses: Float[Array, " component"],
        element_names: Sequence[str],
        element_composition: Int[Array, "element component"],
        *,
        charges: Int[Array, " component"] | None = None,
        provenance: str = "user-supplied",
    ) -> None:
        names = tuple(str(name) for name in component_names)
        elements = tuple(str(name) for name in element_names)
        masses_np = np.asarray(molar_masses, dtype=float)
        composition_np = np.asarray(element_composition)
        charges_np = (
            np.zeros((len(names),), dtype=np.int32)
            if charges is None
            else np.asarray(charges)
        )
        source = str(provenance)

        if not names or any(not name for name in names):
            raise ValueError("component_names must contain non-empty names.")
        if len(set(names)) != len(names):
            raise ValueError("component_names must be unique.")
        if any(not name for name in elements):
            raise ValueError("element_names must contain non-empty names.")
        if len(set(elements)) != len(elements):
            raise ValueError("element_names must be unique.")
        if masses_np.shape != (len(names),):
            raise ValueError("molar_masses must have shape (component_count,).")
        if not np.all(np.isfinite(masses_np)) or np.any(masses_np <= 0.0):
            raise ValueError("molar_masses must be finite and strictly positive.")
        if composition_np.shape != (len(elements), len(names)):
            raise ValueError(
                "element_composition must have shape (element_count, component_count)."
            )
        if not np.issubdtype(composition_np.dtype, np.integer):
            raise TypeError("element_composition must have integer dtype.")
        if np.any(composition_np < 0):
            raise ValueError("element_composition must be nonnegative.")
        if charges_np.shape != (len(names),):
            raise ValueError("charges must have shape (component_count,).")
        if not np.issubdtype(charges_np.dtype, np.integer):
            raise ValueError("charges must have integer dtype.")
        if not source:
            raise ValueError("provenance must be non-empty.")

        masses = jnp.asarray(masses_np)
        composition = jnp.asarray(composition_np, dtype=jnp.int32)
        charge_values = jnp.asarray(charges_np, dtype=jnp.int32)
        content = array_tree_fingerprint(
            {
                "molar_masses": masses_np,
                "element_composition": composition_np,
                "charges": charges_np,
            }
        )

        object.__setattr__(self, "component_names", names)
        object.__setattr__(self, "molar_masses", masses)
        object.__setattr__(self, "element_names", elements)
        object.__setattr__(self, "element_composition", composition)
        object.__setattr__(self, "charges", charge_values)
        object.__setattr__(self, "provenance", source)
        object.__setattr__(self, "component_count", len(names))
        object.__setattr__(self, "element_count", len(elements))
        object.__setattr__(
            self,
            "catalog_id",
            canonical_fingerprint(
                {
                    "kind": "chemical_component_catalog",
                    "component_names": list(names),
                    "element_names": list(elements),
                    "provenance": source,
                    "content": content,
                }
            ),
        )


__all__ = ["ChemicalComponentCatalog"]

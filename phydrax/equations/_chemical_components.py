#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._model import register_artifact_value
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import (
    Dim,
    Float64,
    HostFloat64,
    HostInteger,
    Identifier,
    Int32,
    parse,
    Scope,
    Size,
)


class _ComponentDim(Dim, minimum=1):
    """Number of chemical components."""


class _ElementDim(Dim):
    """Number of chemical elements."""


class ChemicalComponentCatalog(StrictModule, NonTrainableState):
    """Canonical chemical identities shared by phase-specific species occurrences."""

    __strict_contract__ = True

    component_names: tuple[str, ...] = eqx.field(static=True)
    molar_masses: Float64[_ComponentDim]
    element_names: tuple[str, ...] = eqx.field(static=True)
    element_composition: Int32[_ElementDim, _ComponentDim]
    charges: Int32[_ComponentDim]
    provenance: str = eqx.field(static=True)
    component_count: Size[_ComponentDim] = eqx.field(static=True)
    element_count: Size[_ElementDim] = eqx.field(static=True)
    catalog_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        component_names: Sequence[str],
        molar_masses: npt.ArrayLike,
        element_names: Sequence[str],
        element_composition: npt.ArrayLike,
        *,
        charges: npt.ArrayLike | None = None,
        provenance: str = "user-supplied",
    ) -> None:
        names = tuple(str(name) for name in component_names)
        elements = tuple(str(name) for name in element_names)
        masses_np = np.asarray(molar_masses, dtype=np.float64)
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
        scope = Scope()
        component_count = parse(
            len(names), Size[_ComponentDim], "component_count", scope=scope
        )
        element_count = parse(
            len(elements), Size[_ElementDim], "element_count", scope=scope
        )
        parse(masses_np, HostFloat64[_ComponentDim], "molar_masses", scope=scope)
        if not np.all(np.isfinite(masses_np)) or np.any(masses_np <= 0.0):
            raise ValueError("molar_masses must be finite and strictly positive.")
        parse(
            composition_np,
            HostInteger[_ElementDim, _ComponentDim],
            "element_composition",
            scope=scope,
        )
        if np.any(composition_np < 0):
            raise ValueError("element_composition must be nonnegative.")
        parse(charges_np, HostInteger[_ComponentDim], "charges", scope=scope)
        if not source:
            raise ValueError("provenance must be non-empty.")

        content = array_tree_fingerprint(
            {
                "molar_masses": masses_np,
                "element_composition": composition_np,
                "charges": charges_np,
            }
        )
        self.component_names = names
        self.molar_masses = jnp.asarray(masses_np)
        self.element_names = elements
        self.element_composition = jnp.asarray(composition_np, dtype=jnp.int32)
        self.charges = jnp.asarray(charges_np, dtype=jnp.int32)
        self.provenance = source
        self.component_count = component_count
        self.element_count = element_count
        self.catalog_id = canonical_fingerprint(
            {
                "kind": "chemical_component_catalog",
                "component_names": list(names),
                "element_names": list(elements),
                "provenance": source,
                "content": content,
            }
        )


register_artifact_value(
    "phydrax.chemistry:ChemicalComponentCatalog",
    ChemicalComponentCatalog,
)


__all__ = ["ChemicalComponentCatalog"]

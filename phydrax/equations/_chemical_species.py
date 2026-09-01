#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ChemicalPhaseKind(StrEnum):
    """Physical measure supporting a chemical species."""

    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    SURFACE = "surface"
    INERT = "inert"


class ChemicalPhaseSpec(StrictModule, NonTrainableState):
    """Static phase measure and standard-state identity."""

    name: str = eqx.field(static=True)
    kind: ChemicalPhaseKind = eqx.field(static=True)
    measure_dimension: int = eqx.field(static=True)
    standard_concentration: float = eqx.field(static=True)
    site_density: float | None = eqx.field(static=True)
    phase_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: ChemicalPhaseKind,
        measure_dimension: int,
        /,
        *,
        standard_concentration: float = 1.0,
        site_density: float | None = None,
        phase_id: str | None = None,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("Chemical phase name must be nonempty.")
        if not isinstance(kind, ChemicalPhaseKind):
            raise TypeError("kind must be ChemicalPhaseKind.")
        dimension = int(measure_dimension)
        if dimension not in (1, 2, 3):
            raise ValueError("measure_dimension must be one, two, or three.")
        concentration = float(standard_concentration)
        if not np.isfinite(concentration) or concentration <= 0.0:
            raise ValueError("standard_concentration must be finite and positive.")
        density = None if site_density is None else float(site_density)
        if density is not None and (not np.isfinite(density) or density <= 0.0):
            raise ValueError("site_density must be finite and positive when provided.")
        if kind is ChemicalPhaseKind.SURFACE and density is None:
            raise ValueError("Surface phases require site_density.")
        generated = canonical_fingerprint(
            {
                "kind": "chemical-phase",
                "name": name_,
                "phase_kind": kind.value,
                "measure_dimension": dimension,
                "standard_concentration": concentration,
                "site_density": density,
            }
        )
        self.name = name_
        self.kind = kind
        self.measure_dimension = dimension
        self.standard_concentration = concentration
        self.site_density = density
        self.phase_id = generated if phase_id is None else str(phase_id)
        if not self.phase_id:
            raise ValueError("phase_id must be nonempty.")


def _default_phase_spec(kind: ChemicalPhaseKind) -> ChemicalPhaseSpec:
    if kind is ChemicalPhaseKind.SURFACE:
        raise ValueError("Surface species require an explicit ChemicalPhaseSpec.")
    return ChemicalPhaseSpec(
        kind.value,
        kind,
        3,
        standard_concentration=1.0,
    )


class ChemicalSpeciesSchema(StrictModule, NonTrainableState):
    """Canonical species ordering, phase membership, elements, and charge."""

    species_names: tuple[str, ...] = eqx.field(static=True)
    phases: tuple[ChemicalPhaseKind, ...] = eqx.field(static=True)
    phase_specs: tuple[ChemicalPhaseSpec, ...] = eqx.field(static=True)
    phase_ids: Array
    molar_masses: Array
    element_names: tuple[str, ...] = eqx.field(static=True)
    element_composition: Array
    charges: Array
    species_count: int = eqx.field(static=True)
    element_count: int = eqx.field(static=True)
    phase_count: int = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_names,
        phases,
        molar_masses: ArrayLike,
        element_names,
        element_composition: ArrayLike,
        charges: ArrayLike,
        /,
        *,
        phase_specs: tuple[ChemicalPhaseSpec, ...] | None = None,
        schema_id: str | None = None,
    ):
        names = tuple(str(value) for value in species_names)
        phase_values = tuple(phases)
        masses = np.asarray(molar_masses, dtype=float)
        elements = tuple(str(value) for value in element_names)
        composition = np.asarray(element_composition)
        charge_values = np.asarray(charges)
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
            or masses.shape != (len(names),)
            or np.any(~np.isfinite(masses))
            or np.any(masses <= 0.0)
        ):
            raise ValueError("Species names and molar masses are invalid.")
        if len(phase_values) != len(names) or any(
            not isinstance(value, ChemicalPhaseKind) for value in phase_values
        ):
            raise TypeError("phases must contain one ChemicalPhaseKind per species.")
        if (
            not elements
            or any(not value for value in elements)
            or len(set(elements)) != len(elements)
            or composition.shape != (len(elements), len(names))
            or not np.issubdtype(composition.dtype, np.integer)
            or np.any(composition < 0)
        ):
            raise ValueError("Element schema/composition is invalid.")
        if charge_values.shape != (len(names),) or not np.issubdtype(
            charge_values.dtype, np.integer
        ):
            raise ValueError("charges must contain one integer per species.")
        unique_phases = tuple(dict.fromkeys(phase_values))
        if phase_specs is None:
            specs = tuple(_default_phase_spec(kind) for kind in unique_phases)
        else:
            specs = tuple(phase_specs)
            if (
                len(specs) != len(unique_phases)
                or any(not isinstance(value, ChemicalPhaseSpec) for value in specs)
                or tuple(value.kind for value in specs) != unique_phases
                or len({value.name for value in specs}) != len(specs)
            ):
                raise ValueError(
                    "phase_specs must uniquely match phases in first-use order."
                )
        phase_index = {value: index for index, value in enumerate(unique_phases)}
        phase_ids = np.asarray(
            [phase_index[value] for value in phase_values], dtype=np.int32
        )
        generated = canonical_fingerprint(
            {
                "kind": "chemical-species-schema",
                "species": list(names),
                "phases": [value.value for value in phase_values],
                "phase_specs": [value.phase_id for value in specs],
                "molar_masses": array_tree_fingerprint(masses),
                "elements": list(elements),
                "composition": array_tree_fingerprint(composition),
                "charges": array_tree_fingerprint(charge_values),
            }
        )
        self.species_names = names
        self.phases = phase_values
        self.phase_specs = specs
        self.phase_ids = jnp.asarray(phase_ids)
        self.molar_masses = jnp.asarray(masses)
        self.element_names = elements
        self.element_composition = jnp.asarray(composition, dtype=jnp.int32)
        self.charges = jnp.asarray(charge_values, dtype=jnp.int32)
        self.species_count = len(names)
        self.element_count = len(elements)
        self.phase_count = len(specs)
        self.schema_id = generated if schema_id is None else str(schema_id)
        if not self.schema_id:
            raise ValueError("schema_id must be nonempty.")

    def phase_mask(self, phase: ChemicalPhaseKind, /) -> Array:
        if not isinstance(phase, ChemicalPhaseKind):
            raise TypeError("phase must be ChemicalPhaseKind.")
        return jnp.asarray(tuple(value is phase for value in self.phases))

    def phase_species_indices(self, phase: ChemicalPhaseKind, /) -> tuple[int, ...]:
        if not isinstance(phase, ChemicalPhaseKind):
            raise TypeError("phase must be ChemicalPhaseKind.")
        return tuple(index for index, value in enumerate(self.phases) if value is phase)

    def element_amount(self, species_amount: ArrayLike, /) -> Array:
        value = jnp.asarray(species_amount)
        if value.ndim < 1 or value.shape[-1] != self.species_count:
            raise ValueError("species_amount must end in species axis.")
        return contract("es,...s->...e", self.element_composition, value)

    def charge_amount(self, species_amount: ArrayLike, /) -> Array:
        value = jnp.asarray(species_amount)
        if value.ndim < 1 or value.shape[-1] != self.species_count:
            raise ValueError("species_amount must end in species axis.")
        return contract("s,...s->...", self.charges, value)


__all__ = [
    "ChemicalPhaseKind",
    "ChemicalPhaseSpec",
    "ChemicalSpeciesSchema",
]

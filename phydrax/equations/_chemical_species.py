#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_components import ChemicalComponentCatalog


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
    standard_pressure: float | None = eqx.field(static=True)
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
        standard_pressure: float | None = None,
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
        pressure = None if standard_pressure is None else float(standard_pressure)
        if pressure is not None and (not np.isfinite(pressure) or pressure <= 0.0):
            raise ValueError("standard_pressure must be finite and positive.")
        if kind is ChemicalPhaseKind.GAS and pressure is None:
            raise ValueError("Gas phases require standard_pressure.")
        if kind is not ChemicalPhaseKind.GAS and pressure is not None:
            raise ValueError("standard_pressure is defined only for gas phases.")
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
                "standard_pressure": pressure,
                "site_density": density,
            }
        )
        self.name = name_
        self.kind = kind
        self.measure_dimension = dimension
        self.standard_concentration = concentration
        self.standard_pressure = pressure
        self.site_density = density
        self.phase_id = generated if phase_id is None else str(phase_id)
        if not self.phase_id:
            raise ValueError("phase_id must be nonempty.")


def _default_phase_spec(
    kind: ChemicalPhaseKind,
    gas_standard_pressure: float | None,
) -> ChemicalPhaseSpec:
    if kind is ChemicalPhaseKind.SURFACE:
        raise ValueError("Surface species require an explicit ChemicalPhaseSpec.")
    return ChemicalPhaseSpec(
        kind.value,
        kind,
        3,
        standard_concentration=1.0,
        standard_pressure=gas_standard_pressure
        if kind is ChemicalPhaseKind.GAS
        else None,
    )


class ChemicalSpeciesSchema(StrictModule, NonTrainableState):
    """Phase-specific species occurrences over canonical chemical components."""

    catalog: ChemicalComponentCatalog
    species_names: tuple[str, ...] = eqx.field(static=True)
    species_component_indices: Array
    phases: tuple[ChemicalPhaseKind, ...] = eqx.field(static=True)
    phase_specs: tuple[ChemicalPhaseSpec, ...] = eqx.field(static=True)
    phase_ids: Array
    molar_masses: Array
    element_names: tuple[str, ...] = eqx.field(static=True)
    element_composition: Array
    charges: Array
    species_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    element_count: int = eqx.field(static=True)
    phase_count: int = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        catalog: ChemicalComponentCatalog,
        species_names,
        species_component_indices: ArrayLike,
        phase_specs: tuple[ChemicalPhaseSpec, ...],
        species_phase_indices: ArrayLike,
        /,
        *,
        schema_id: str | None = None,
    ):
        if not isinstance(catalog, ChemicalComponentCatalog):
            raise TypeError("catalog must be ChemicalComponentCatalog.")
        names = tuple(str(value) for value in species_names)
        component_indices = np.asarray(species_component_indices)
        specs = tuple(phase_specs)
        phase_indices = np.asarray(species_phase_indices)
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("species_names must be non-empty and unique.")
        if component_indices.shape != (len(names),) or not np.issubdtype(
            component_indices.dtype, np.integer
        ):
            raise TypeError(
                "species_component_indices must contain one integer per species."
            )
        if np.any(component_indices < 0) or np.any(
            component_indices >= catalog.component_count
        ):
            raise ValueError("species_component_indices contains an invalid index.")
        if (
            not specs
            or any(not isinstance(value, ChemicalPhaseSpec) for value in specs)
            or len({value.name for value in specs}) != len(specs)
            or len({value.phase_id for value in specs}) != len(specs)
        ):
            raise ValueError("phase_specs must contain unique named phase instances.")
        if phase_indices.shape != (len(names),) or not np.issubdtype(
            phase_indices.dtype, np.integer
        ):
            raise TypeError("species_phase_indices must contain one integer per species.")
        if np.any(phase_indices < 0) or np.any(phase_indices >= len(specs)):
            raise ValueError("species_phase_indices contains an invalid index.")

        component_indices = component_indices.astype(np.int32, copy=False)
        phase_indices = phase_indices.astype(np.int32, copy=False)
        phases = tuple(specs[index].kind for index in phase_indices)
        masses = np.asarray(catalog.molar_masses)[component_indices]
        composition = np.asarray(catalog.element_composition)[:, component_indices]
        charges = np.asarray(catalog.charges)[component_indices]
        generated = canonical_fingerprint(
            {
                "kind": "chemical-species-schema",
                "catalog_id": catalog.catalog_id,
                "species": list(names),
                "component_indices": array_tree_fingerprint(component_indices),
                "phase_specs": [value.phase_id for value in specs],
                "phase_indices": array_tree_fingerprint(phase_indices),
            }
        )
        self.catalog = catalog
        self.species_names = names
        self.species_component_indices = jnp.asarray(component_indices)
        self.phases = phases
        self.phase_specs = specs
        self.phase_ids = jnp.asarray(phase_indices)
        self.molar_masses = jnp.asarray(masses)
        self.element_names = catalog.element_names
        self.element_composition = jnp.asarray(composition, dtype=jnp.int32)
        self.charges = jnp.asarray(charges, dtype=jnp.int32)
        self.species_count = len(names)
        self.component_count = catalog.component_count
        self.element_count = catalog.element_count
        self.phase_count = len(specs)
        self.schema_id = generated if schema_id is None else str(schema_id)
        if not self.schema_id:
            raise ValueError("schema_id must be nonempty.")

    @classmethod
    def from_unique_species(
        cls,
        species_names,
        phases,
        molar_masses: ArrayLike,
        element_names,
        element_composition: ArrayLike,
        charges: ArrayLike,
        /,
        *,
        phase_specs: tuple[ChemicalPhaseSpec, ...] | None = None,
        gas_standard_pressure: float | None = None,
        schema_id: str | None = None,
        provenance: str = "user-supplied",
    ) -> ChemicalSpeciesSchema:
        """Construct the common one-component-per-species schema."""
        names = tuple(str(value) for value in species_names)
        phase_values = tuple(phases)
        if len(phase_values) != len(names) or any(
            not isinstance(value, ChemicalPhaseKind) for value in phase_values
        ):
            raise TypeError("phases must contain one ChemicalPhaseKind per species.")
        unique_phases = tuple(dict.fromkeys(phase_values))
        if phase_specs is None:
            if ChemicalPhaseKind.GAS in unique_phases and gas_standard_pressure is None:
                raise ValueError("Gas species require explicit gas_standard_pressure.")
            specs = tuple(
                _default_phase_spec(kind, gas_standard_pressure) for kind in unique_phases
            )
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
        catalog = ChemicalComponentCatalog(
            names,
            molar_masses,
            element_names,
            element_composition,
            charges=charges,
            provenance=provenance,
        )
        return cls(
            catalog,
            names,
            np.arange(len(names), dtype=np.int32),
            specs,
            np.asarray([phase_index[value] for value in phase_values], dtype=np.int32),
            schema_id=schema_id,
        )

    def phase_mask(self, phase: ChemicalPhaseKind, /) -> Array:
        if not isinstance(phase, ChemicalPhaseKind):
            raise TypeError("phase must be ChemicalPhaseKind.")
        return jnp.asarray(tuple(value is phase for value in self.phases))

    def phase_slot_mask(self, phase_index: int, /) -> Array:
        index = int(phase_index)
        if index < 0 or index >= self.phase_count:
            raise ValueError("phase_index is outside the phase schema.")
        return self.phase_ids == index

    def phase_species_indices(self, phase: ChemicalPhaseKind, /) -> tuple[int, ...]:
        if not isinstance(phase, ChemicalPhaseKind):
            raise TypeError("phase must be ChemicalPhaseKind.")
        return tuple(index for index, value in enumerate(self.phases) if value is phase)

    def phase_slot_species_indices(self, phase_index: int, /) -> tuple[int, ...]:
        index = int(phase_index)
        if index < 0 or index >= self.phase_count:
            raise ValueError("phase_index is outside the phase schema.")
        return tuple(
            species_index
            for species_index, value in enumerate(np.asarray(self.phase_ids))
            if value == index
        )

    def component_amount(self, species_amount: ArrayLike, /) -> Array:
        value = jnp.asarray(species_amount)
        if value.ndim < 1 or value.shape[-1] != self.species_count:
            raise ValueError("species_amount must end in species axis.")
        incidence = jax.nn.one_hot(
            self.species_component_indices,
            self.component_count,
            dtype=value.dtype,
        )
        return contract("sc,...s->...c", incidence, value)

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
    "ChemicalComponentCatalog",
    "ChemicalPhaseKind",
    "ChemicalPhaseSpec",
    "ChemicalSpeciesSchema",
]

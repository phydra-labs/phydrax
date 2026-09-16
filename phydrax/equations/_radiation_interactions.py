#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import (
    conversion_factor,
    derived_unit,
    ELECTRONVOLT,
    JOULE,
    KILOGRAM,
    METER,
    UnitDefinition,
)
from ._diagnostic_photon import (
    DiagnosticPhotonCoefficientRole,
    DiagnosticPhotonCoefficientTable,
    DiagnosticPhotonInterpolationPolicy,
)


_MASS_ATTENUATION_SI = derived_unit("m2/kg", ((METER, 2), (KILOGRAM, -1)))


class RadiationInteractionKind(IntEnum):
    PHOTOELECTRIC = 0
    COMPTON = 1
    RAYLEIGH = 2


class RadiationCrossSectionEvaluation(StrictModule):
    coefficients: Array
    total: Array
    supported: Array
    finite: Array
    successful: Array
    library_id: str = eqx.field(static=True)


class RadiationCrossSectionLibrary(StrictModule, NonTrainableState):
    """Macroscopic photon interaction coefficients on one governed energy grid."""

    energy: Array
    coefficients: Array
    mass_density_kg_per_m3: Array
    log_interpolation: Array
    material_ids: tuple[str, ...] = eqx.field(static=True)
    process_ids: tuple[str, str, str] = eqx.field(static=True)
    energy_unit: UnitDefinition = eqx.field(static=True)
    source_table_ids: tuple[str, str, str] = eqx.field(static=True)
    source_provenance_ids: tuple[str, str, str] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        photoelectric: DiagnosticPhotonCoefficientTable,
        compton: DiagnosticPhotonCoefficientTable,
        rayleigh: DiagnosticPhotonCoefficientTable,
        mass_density_kg_per_m3: ArrayLike,
        /,
        *,
        commercial_use: bool = False,
        redistribution: bool = False,
        export: bool = False,
    ):
        tables = (photoelectric, compton, rayleigh)
        if not all(
            isinstance(table, DiagnosticPhotonCoefficientTable) for table in tables
        ):
            raise TypeError(
                "Photon interactions require three DiagnosticPhotonCoefficientTable "
                "values."
            )
        if any(
            table.role is not DiagnosticPhotonCoefficientRole.MASS_ATTENUATION
            for table in tables
        ):
            raise ValueError(
                "Photoelectric, Compton, and Rayleigh inputs must be mass-attenuation "
                "tables."
            )
        material_ids = photoelectric.material_ids
        energy_grid_id = photoelectric.energy_grid.grid_id
        if any(table.material_ids != material_ids for table in tables[1:]):
            raise ValueError(
                "Photon interaction tables must use one exact ordered material basis."
            )
        if any(table.energy_grid.grid_id != energy_grid_id for table in tables[1:]):
            raise ValueError(
                "Photon interaction tables must use one exact photon energy grid."
            )
        if len({table.table_id for table in tables}) != len(tables):
            raise ValueError(
                "Photon interaction process tables must have distinct identities."
            )
        for table in tables:
            table.provenance.reference.require_rights(
                commercial_use=commercial_use,
                redistribution=redistribution,
                export=export,
            )
        density = np.asarray(mass_density_kg_per_m3, dtype=float)
        if (
            density.shape != (len(material_ids),)
            or np.any(~np.isfinite(density))
            or np.any(density <= 0.0)
        ):
            raise ValueError(
                "mass_density_kg_per_m3 must be finite and positive per material."
            )
        energy = np.asarray(photoelectric.energy_grid.energy_j, dtype=float) * float(
            conversion_factor(JOULE, ELECTRONVOLT)
        )
        mass_coefficients = np.stack(
            tuple(
                np.asarray(table.values, dtype=float)
                * float(conversion_factor(table.unit, _MASS_ATTENUATION_SI))
                for table in tables
            ),
            axis=-1,
        )
        coefficients = density[:, None, None] * mass_coefficients
        if np.any(np.sum(coefficients, axis=-1) <= 0.0):
            raise ValueError(
                "Every material/energy point requires positive total attenuation."
            )
        source_table_ids = tuple(table.table_id for table in tables)
        source_provenance_ids = tuple(table.provenance.provenance_id for table in tables)
        log_interpolation = np.asarray(
            tuple(
                table.interpolation is DiagnosticPhotonInterpolationPolicy.LOG_LOG
                for table in tables
            ),
            dtype=bool,
        )
        self.energy = jnp.asarray(energy)
        self.coefficients = jnp.asarray(coefficients)
        self.mass_density_kg_per_m3 = jnp.asarray(density)
        self.log_interpolation = jnp.asarray(log_interpolation)
        self.material_ids = material_ids
        self.process_ids = ("photoelectric", "compton", "rayleigh")
        self.energy_unit = ELECTRONVOLT
        self.source_table_ids = source_table_ids
        self.source_provenance_ids = source_provenance_ids
        self.library_id = canonical_fingerprint(
            {
                "kind": "prepared-governed-photon-interaction-library",
                "energy": array_tree_fingerprint(energy),
                "coefficients": array_tree_fingerprint(coefficients),
                "mass_density_kg_per_m3": array_tree_fingerprint(density),
                "log_interpolation": array_tree_fingerprint(log_interpolation),
                "materials": material_ids,
                "processes": self.process_ids,
                "energy_unit": ELECTRONVOLT.unit_id,
                "source_tables": source_table_ids,
                "source_provenance": source_provenance_ids,
            }
        )

    @property
    def material_count(self) -> int:
        return len(self.material_ids)

    def evaluate(
        self, material_index: ArrayLike, photon_energy: ArrayLike, /
    ) -> RadiationCrossSectionEvaluation:
        material = jnp.asarray(material_index, dtype=jnp.int32)
        energy = jnp.asarray(photon_energy, dtype=self.energy.dtype)
        shape = jnp.broadcast_shapes(material.shape, energy.shape)
        material = jnp.broadcast_to(material, shape)
        energy = jnp.broadcast_to(energy, shape)
        supported = (
            (material >= 0)
            & (material < self.material_count)
            & jnp.isfinite(energy)
            & (energy >= self.energy[0])
            & (energy <= self.energy[-1])
        )
        safe_material = jnp.clip(material, 0, self.material_count - 1)
        safe_energy = jnp.clip(energy, self.energy[0], self.energy[-1])
        upper = jnp.searchsorted(self.energy, safe_energy, side="right")
        upper = jnp.clip(upper, 1, self.energy.size - 1)
        lower = upper - 1
        left_energy = self.energy[lower]
        right_energy = self.energy[upper]
        linear_fraction = (safe_energy - left_energy) / (right_energy - left_energy)
        log_fraction = (jnp.log(safe_energy) - jnp.log(left_energy)) / (
            jnp.log(right_energy) - jnp.log(left_energy)
        )
        left = self.coefficients[safe_material, lower]
        right = self.coefficients[safe_material, upper]
        linear_evaluated = left + linear_fraction[..., None] * (right - left)
        tiny = jnp.finfo(left.dtype).tiny
        log_evaluated = jnp.exp(
            jnp.log(jnp.maximum(left, tiny))
            + log_fraction[..., None]
            * (jnp.log(jnp.maximum(right, tiny)) - jnp.log(jnp.maximum(left, tiny)))
        )
        evaluated = jnp.where(self.log_interpolation, log_evaluated, linear_evaluated)
        total = jnp.sum(evaluated, axis=-1)
        finite = jnp.all(jnp.isfinite(evaluated), axis=-1) & jnp.isfinite(total)
        successful = (
            supported & finite & jnp.all(evaluated >= 0.0, axis=-1) & (total > 0.0)
        )
        return RadiationCrossSectionEvaluation(
            evaluated, total, supported, finite, successful, self.library_id
        )

    def majorant(self, photon_energy: ArrayLike, /) -> Array:
        energy = jnp.asarray(photon_energy, dtype=self.energy.dtype)
        material = jnp.arange(self.material_count, dtype=jnp.int32)
        evaluated = self.evaluate(
            material.reshape((self.material_count,) + (1,) * energy.ndim),
            energy[None, ...],
        )
        return jnp.max(evaluated.total, axis=0)


__all__ = [
    "RadiationCrossSectionEvaluation",
    "RadiationCrossSectionLibrary",
    "RadiationInteractionKind",
]

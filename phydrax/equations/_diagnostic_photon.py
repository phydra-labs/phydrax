#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-pinned diagnostic-photon material coefficient tables."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..nuclear._provenance import NuclearDataProvenance
from ..units import AREA, MASS, UnitDefinition


class DiagnosticPhotonCoefficientRole(StrEnum):
    """Physical role of a diagnostic-photon mass coefficient."""

    MASS_ATTENUATION = "mass_attenuation"
    MASS_ENERGY_TRANSFER = "mass_energy_transfer"
    MASS_ENERGY_ABSORPTION = "mass_energy_absorption"


class DiagnosticPhotonInterpolationPolicy(StrEnum):
    """Declared interpolation coordinates for a coefficient table."""

    LINEAR = "linear"
    LOG_LOG = "log_log"


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return normalized


class PhotonEnergyGrid(StrictModule, NonTrainableState):
    """Immutable positive, strictly increasing photon energies in joules."""

    energy_j: Array
    grid_id: str = eqx.field(static=True)

    def __init__(self, energy_j: ArrayLike, /) -> None:
        values = np.array(energy_j, dtype=np.float64, copy=True)
        if (
            values.ndim != 1
            or values.size < 2
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or np.any(np.diff(values) <= 0.0)
        ):
            raise ValueError(
                "Photon energies must be a finite, positive, strictly increasing vector."
            )
        self.energy_j = jnp.asarray(values)
        self.grid_id = canonical_fingerprint(
            {
                "kind": "photon-energy-grid",
                "energy_j": array_tree_fingerprint(values),
            }
        )


class DiagnosticPhotonInterpolationEvidence(StrictModule, NonTrainableState):
    """Bracketing, weighting, and support evidence for one bounded evaluation."""

    query_energy_j: Array
    lower_indices: Array
    upper_indices: Array
    upper_weights: Array
    supported: Array
    interpolated: Array
    interpolation: DiagnosticPhotonInterpolationPolicy = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class DiagnosticPhotonCoefficientEvaluation(StrictModule, NonTrainableState):
    """Coefficients on an exact ordered material basis with audit evidence."""

    coefficient: Array
    evidence: DiagnosticPhotonInterpolationEvidence
    unit: UnitDefinition
    role: DiagnosticPhotonCoefficientRole = eqx.field(static=True)
    material_ids: tuple[str, ...] = eqx.field(static=True)
    table_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class DiagnosticPhotonCoefficientTable(StrictModule, NonTrainableState):
    """Bounded diagnostic-photon coefficients on an ordered material basis."""

    role: DiagnosticPhotonCoefficientRole = eqx.field(static=True)
    energy_grid: PhotonEnergyGrid
    material_ids: tuple[str, ...] = eqx.field(static=True)
    values: Array
    unit: UnitDefinition
    provenance: NuclearDataProvenance
    interpolation: DiagnosticPhotonInterpolationPolicy = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        role: DiagnosticPhotonCoefficientRole,
        energy_grid: PhotonEnergyGrid,
        material_ids: Sequence[str],
        values: ArrayLike,
        unit: UnitDefinition,
        provenance: NuclearDataProvenance,
        interpolation: DiagnosticPhotonInterpolationPolicy | str,
        /,
    ) -> None:
        if not isinstance(role, DiagnosticPhotonCoefficientRole):
            raise TypeError("role must be DiagnosticPhotonCoefficientRole.")
        if not isinstance(energy_grid, PhotonEnergyGrid):
            raise TypeError("energy_grid must be PhotonEnergyGrid.")
        identifiers = tuple(_text(value, "material_id") for value in material_ids)
        if not identifiers:
            raise ValueError("material_ids must contain at least one material identity.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("material_ids must be unique and explicitly ordered.")
        coefficients = np.array(values, dtype=np.float64, copy=True)
        if coefficients.shape != (len(identifiers), energy_grid.energy_j.size):
            raise ValueError(
                "values must have shape (number of materials, number of photon energies)."
            )
        if np.any(~np.isfinite(coefficients)) or np.any(coefficients < 0.0):
            raise ValueError(
                "Diagnostic photon coefficients must be finite and nonnegative."
            )
        if not isinstance(unit, UnitDefinition):
            raise TypeError("unit must be UnitDefinition.")
        if unit.dimension != AREA / MASS:
            raise ValueError(
                "Diagnostic photon mass coefficients require an area-per-mass unit."
            )
        if not isinstance(provenance, NuclearDataProvenance):
            raise TypeError("provenance must be NuclearDataProvenance.")
        if isinstance(interpolation, str):
            if interpolation not in {
                value.value for value in DiagnosticPhotonInterpolationPolicy
            }:
                raise ValueError("interpolation must be 'linear' or 'log_log'.")
            interpolation_policy = DiagnosticPhotonInterpolationPolicy(interpolation)
        else:
            if not isinstance(interpolation, DiagnosticPhotonInterpolationPolicy):
                raise TypeError(
                    "interpolation must be DiagnosticPhotonInterpolationPolicy "
                    "or its canonical string value."
                )
            interpolation_policy = interpolation
        if interpolation_policy is DiagnosticPhotonInterpolationPolicy.LOG_LOG and np.any(
            coefficients <= 0.0
        ):
            raise ValueError("Log-log interpolation requires strictly positive values.")
        self.role = role
        self.energy_grid = energy_grid
        self.material_ids = identifiers
        self.values = jnp.asarray(coefficients)
        self.unit = unit
        self.provenance = provenance
        self.interpolation = interpolation_policy
        self.table_id = canonical_fingerprint(
            {
                "kind": "diagnostic-photon-coefficient-table",
                "role": role.value,
                "energy_grid": energy_grid.grid_id,
                "material_ids": list(identifiers),
                "values": array_tree_fingerprint(coefficients),
                "unit": unit.unit_id,
                "provenance": provenance.provenance_id,
                "interpolation": interpolation_policy.value,
            }
        )

    def evaluate(
        self,
        energy_j: ArrayLike,
        material_ids: Sequence[str],
        /,
        *,
        expected_provenance: NuclearDataProvenance | None = None,
    ) -> DiagnosticPhotonCoefficientEvaluation:
        """Interpolate with support evidence after checking basis and source pin."""
        requested_materials = tuple(_text(value, "material_id") for value in material_ids)
        if requested_materials != self.material_ids:
            raise ValueError(
                "Requested material IDs must exactly match the table's ordered basis."
            )
        if expected_provenance is not None:
            if not isinstance(expected_provenance, NuclearDataProvenance):
                raise TypeError(
                    "expected_provenance must be NuclearDataProvenance or None."
                )
            if expected_provenance.provenance_id != self.provenance.provenance_id:
                raise ValueError(
                    "Expected nuclear-data provenance does not match the "
                    "coefficient table."
                )
        query_raw = jnp.asarray(energy_j)
        if jnp.issubdtype(query_raw.dtype, jnp.complexfloating):
            raise TypeError("Photon energy queries must be real-valued.")
        query = query_raw.astype(jnp.result_type(query_raw, float))
        grid = self.energy_grid.energy_j.astype(query.dtype)
        finite_positive = jnp.isfinite(query) & (query > 0.0)
        supported = finite_positive & (query >= grid[0]) & (query <= grid[-1])
        safe_query = jnp.clip(
            jnp.where(finite_positive, query, grid[0]),
            grid[0],
            grid[-1],
        )
        lower = jnp.clip(
            jnp.searchsorted(grid, safe_query, side="right") - 1,
            0,
            int(grid.size) - 2,
        ).astype(jnp.int32)
        upper = lower + 1
        lower_energy = grid[lower]
        upper_energy = grid[upper]
        if self.interpolation is DiagnosticPhotonInterpolationPolicy.LINEAR:
            upper_weight = (safe_query - lower_energy) / (upper_energy - lower_energy)
            lower_values = self.values[:, lower]
            upper_values = self.values[:, upper]
            coefficient = lower_values + upper_weight * (upper_values - lower_values)
        else:
            upper_weight = (jnp.log(safe_query) - jnp.log(lower_energy)) / (
                jnp.log(upper_energy) - jnp.log(lower_energy)
            )
            lower_values = self.values[:, lower]
            upper_values = self.values[:, upper]
            coefficient = jnp.exp(
                jnp.log(lower_values)
                + upper_weight * (jnp.log(upper_values) - jnp.log(lower_values))
            )
        coefficient = jnp.where(supported[jnp.newaxis, ...], coefficient, 0.0)
        interpolated = (
            supported & (safe_query != lower_energy) & (safe_query != upper_energy)
        )
        evidence = DiagnosticPhotonInterpolationEvidence(
            query,
            lower,
            upper,
            upper_weight,
            supported,
            interpolated,
            self.interpolation,
            self.energy_grid.grid_id,
            self.provenance.provenance_id,
        )
        return DiagnosticPhotonCoefficientEvaluation(
            coefficient,
            evidence,
            self.unit,
            self.role,
            self.material_ids,
            self.table_id,
            self.provenance.provenance_id,
        )


__all__ = [
    "DiagnosticPhotonCoefficientEvaluation",
    "DiagnosticPhotonCoefficientRole",
    "DiagnosticPhotonCoefficientTable",
    "DiagnosticPhotonInterpolationEvidence",
    "DiagnosticPhotonInterpolationPolicy",
    "PhotonEnergyGrid",
]

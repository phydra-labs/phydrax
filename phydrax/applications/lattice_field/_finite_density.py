#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum, StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import ENERGY, UnitDefinition


class FiniteDensitySourceKind(StrEnum):
    FINITE_REGULATOR = "finite-regulator-lattice"
    CONTINUUM_EXTRAPOLATED = "continuum-extrapolated-lattice"
    IMAGINARY_CHEMICAL_POTENTIAL = "imaginary-chemical-potential"
    REWEIGHTED = "reweighted"
    RESUMMED = "resummed-lattice-construction"
    HRG = "phenomenological-hrg"
    CRITICAL_MODEL = "phenomenological-3d-ising-critical-eos"


class FiniteDensityStatus(IntEnum):
    SUCCESS = 0
    OUTSIDE_DOMAIN = 1
    NONFINITE = 2
    INCOMPLETE = 3
    UNQUALIFIED_SOURCE = 4
    CONSTRAINT_FAILURE = 5
    UNSTABLE = 6


class ChemicalChargeConvention(StrictModule, NonTrainableState):
    charges: tuple[str, str, str] = eqx.field(static=True)
    chemical_potential_kind: str = eqx.field(static=True)
    pressure_normalization: str = eqx.field(static=True)
    energy_unit: UnitDefinition
    cp_symmetric: bool = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        energy_unit: UnitDefinition,
        chemical_potential_kind: str = "baryon-electric-strangeness",
        pressure_normalization: str = "p-over-T4",
        cp_symmetric: bool = True,
    ):
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise ValueError("energy_unit must have energy dimension.")
        kind = str(chemical_potential_kind).strip()
        normalization = str(pressure_normalization).strip()
        if kind != "baryon-electric-strangeness" or normalization != "p-over-T4":
            raise ValueError(
                "Only explicit B/Q/S chemical potentials and p/T^4 are supported."
            )
        self.charges = ("B", "Q", "S")
        self.chemical_potential_kind = kind
        self.pressure_normalization = normalization
        self.energy_unit = energy_unit
        self.cp_symmetric = bool(cp_symmetric)
        self.convention_id = canonical_fingerprint(
            {
                "kind": "finite-density-charge-convention",
                "charges": list(self.charges),
                "chemical_potential_kind": kind,
                "pressure_normalization": normalization,
                "energy_unit": energy_unit.unit_id,
                "cp_symmetric": bool(cp_symmetric),
            }
        )


class GeneralizedSusceptibilityIndex(StrictModule, NonTrainableState):
    orders: tuple[int, int, int] = eqx.field(static=True)
    total_order: int = eqx.field(static=True)
    label: str = eqx.field(static=True)
    index_id: str = eqx.field(static=True)

    def __init__(self, baryon_order: int, charge_order: int, strangeness_order: int, /):
        orders = tuple(map(int, (baryon_order, charge_order, strangeness_order)))
        if any(value < 0 for value in orders):
            raise ValueError("Susceptibility derivative orders must be nonnegative.")
        total = sum(orders)
        label = f"chi_B{orders[0]}_Q{orders[1]}_S{orders[2]}"
        self.orders = orders
        self.total_order = total
        self.label = label
        self.index_id = canonical_fingerprint(
            {"kind": "generalized-bqs-susceptibility-index", "orders": list(orders)}
        )


class FiniteDensityDomain(StrictModule, NonTrainableState):
    temperature_bounds: tuple[float, float] = eqx.field(static=True)
    chemical_potential_over_temperature_bounds: tuple[tuple[float, float], ...] = (
        eqx.field(static=True)
    )
    maximum_total_order: int = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperature_bounds: tuple[float, float],
        chemical_potential_over_temperature_bounds: Sequence[tuple[float, float]],
        /,
        *,
        maximum_total_order: int,
    ):
        temperatures = tuple(map(float, temperature_bounds))
        bounds = tuple(
            tuple(map(float, item)) for item in chemical_potential_over_temperature_bounds
        )
        maximum = int(maximum_total_order)
        if len(bounds) != 3 or not 0.0 < temperatures[0] < temperatures[1] or maximum < 0:
            raise ValueError(
                "Finite-density domain shape, temperature, or order is invalid."
            )
        if any(
            len(item) != 2
            or not math.isfinite(item[0])
            or not math.isfinite(item[1])
            or item[0] >= item[1]
            for item in bounds
        ):
            raise ValueError(
                "Every chemical-potential ratio bound must be finite and ordered."
            )
        self.temperature_bounds = temperatures
        self.chemical_potential_over_temperature_bounds = bounds
        self.maximum_total_order = maximum
        self.domain_id = canonical_fingerprint(
            {
                "kind": "finite-density-domain",
                "temperature_bounds": list(temperatures),
                "chemical_potential_over_temperature_bounds": [
                    list(value) for value in bounds
                ],
                "maximum_total_order": maximum,
            }
        )

    def contains(
        self, temperature: ArrayLike, chemical_potential_over_temperature: ArrayLike, /
    ) -> Array:
        temperature_ = jnp.asarray(temperature)
        ratios = jnp.asarray(
            chemical_potential_over_temperature, dtype=temperature_.dtype
        )
        if ratios.shape[-1:] != (3,):
            raise ValueError("Chemical-potential ratios require trailing B/Q/S axis.")
        lower = jnp.asarray(
            [value[0] for value in self.chemical_potential_over_temperature_bounds]
        )
        upper = jnp.asarray(
            [value[1] for value in self.chemical_potential_over_temperature_bounds]
        )
        return (
            (temperature_ >= self.temperature_bounds[0])
            & (temperature_ <= self.temperature_bounds[1])
            & jnp.all((ratios >= lower) & (ratios <= upper), axis=-1)
        )


class SusceptibilityEstimate(StrictModule, NonTrainableState):
    temperatures: Array
    values: Array
    covariance: Array
    valid: Array
    indices: tuple[GeneralizedSusceptibilityIndex, ...]
    convention: ChemicalChargeConvention
    domain: FiniteDensityDomain
    source_kind: FiniteDensitySourceKind = eqx.field(static=True)
    provenance_ids: tuple[str, ...] = eqx.field(static=True)
    estimate_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperatures: ArrayLike,
        values: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        indices: Sequence[GeneralizedSusceptibilityIndex],
        convention: ChemicalChargeConvention,
        domain: FiniteDensityDomain,
        source_kind: FiniteDensitySourceKind,
        provenance_ids: Sequence[str],
    ):
        temperatures_ = np.asarray(temperatures, dtype=np.float64)
        values_ = np.asarray(values, dtype=np.float64)
        covariance_ = np.asarray(covariance, dtype=np.float64)
        indices_ = tuple(indices)
        if (
            temperatures_.ndim != 1
            or temperatures_.size < 2
            or np.any(~np.isfinite(temperatures_))
            or np.any(np.diff(temperatures_) <= 0.0)
        ):
            raise ValueError("Temperatures must be a finite increasing vector.")
        if (
            not indices_
            or any(
                not isinstance(value, GeneralizedSusceptibilityIndex)
                for value in indices_
            )
            or len({value.index_id for value in indices_}) != len(indices_)
        ):
            raise ValueError("Susceptibility indices must be distinct typed values.")
        expected = (temperatures_.size, len(indices_))
        if values_.shape != expected or covariance_.shape != (values_.size, values_.size):
            raise ValueError(
                "Susceptibility values or joint covariance shape is invalid."
            )
        if (
            not isinstance(convention, ChemicalChargeConvention)
            or not isinstance(domain, FiniteDensityDomain)
            or not isinstance(source_kind, FiniteDensitySourceKind)
        ):
            raise TypeError(
                "Finite-density convention, domain, and source kind must be explicit."
            )
        if any(value.total_order > domain.maximum_total_order for value in indices_):
            raise ValueError("A susceptibility index exceeds the domain's maximum order.")
        valid_ = (
            np.ones(expected, dtype=np.bool_)
            if valid is None
            else np.asarray(valid, dtype=np.bool_)
        )
        if valid_.shape != expected:
            raise ValueError("valid must align with susceptibility values.")
        provenance = tuple(str(value).strip() for value in provenance_ids)
        if (
            not provenance
            or any(not value for value in provenance)
            or len(set(provenance)) != len(provenance)
        ):
            raise ValueError("Distinct provenance_ids are required.")
        if np.any(~np.isfinite(covariance_)) or not np.allclose(
            covariance_, covariance_.T
        ):
            raise ValueError("Susceptibility covariance must be finite and symmetric.")
        eigenvalues = np.linalg.eigvalsh(covariance_)
        if np.min(eigenvalues) < -1.0e-10 * max(np.max(np.abs(eigenvalues)), 1.0):
            raise ValueError("Susceptibility covariance must be positive semidefinite.")
        if convention.cp_symmetric:
            odd = np.asarray([index.total_order % 2 == 1 for index in indices_])
            if np.any(np.abs(values_[:, odd]) > 256.0 * np.finfo(values_.dtype).eps):
                raise ValueError(
                    "CP-symmetric susceptibility input contains nonzero odd total orders."
                )
        self.temperatures = jnp.asarray(temperatures_)
        self.values = jnp.asarray(values_)
        self.covariance = jnp.asarray(covariance_)
        self.valid = jnp.asarray(valid_)
        self.indices = indices_
        self.convention = convention
        self.domain = domain
        self.source_kind = source_kind
        self.provenance_ids = provenance
        self.estimate_id = canonical_fingerprint(
            {
                "kind": "bqs-susceptibility-estimate",
                "temperatures": array_tree_fingerprint(temperatures_),
                "values": array_tree_fingerprint(values_),
                "covariance": array_tree_fingerprint(covariance_),
                "valid": array_tree_fingerprint(valid_),
                "indices": [value.index_id for value in indices_],
                "convention": convention.convention_id,
                "domain": domain.domain_id,
                "source_kind": source_kind.value,
                "provenance": list(provenance),
            }
        )


__all__ = [
    "ChemicalChargeConvention",
    "FiniteDensityDomain",
    "FiniteDensitySourceKind",
    "FiniteDensityStatus",
    "GeneralizedSusceptibilityIndex",
    "SusceptibilityEstimate",
]

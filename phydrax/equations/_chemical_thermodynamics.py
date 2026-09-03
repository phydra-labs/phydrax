#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema


UNIVERSAL_GAS_CONSTANT = 8.31446261815324


class SpeciesThermodynamicEvaluation(StrictModule):
    molar_heat_capacity_pressure: Array
    molar_heat_capacity_volume: Array
    molar_enthalpy: Array
    molar_internal_energy: Array
    molar_entropy: Array
    molar_gibbs_energy: Array
    active_interval: Array
    temperature_margin: Array
    successful: Array
    thermodynamics_id: str = eqx.field(static=True)


class AbstractSpeciesThermodynamicsPlan(StrictModule, NonTrainableState, abc.ABC):
    schema: ChemicalSpeciesSchema
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    thermodynamics_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def evaluate(self, temperature: ArrayLike, /) -> SpeciesThermodynamicEvaluation:
        raise NotImplementedError


class PolynomialSpeciesThermodynamicsPlan(AbstractSpeciesThermodynamicsPlan):
    """Species internal-energy polynomials with thermodynamic completion."""

    heat_capacity_volume_coefficients: Array
    reference_molar_internal_energy: Array
    reference_molar_entropy: Array
    reference_temperature: float = eqx.field(static=True)

    def __init__(
        self,
        schema: ChemicalSpeciesSchema,
        heat_capacity_volume: ArrayLike,
        reference_molar_internal_energy: ArrayLike,
        /,
        *,
        reference_molar_entropy: ArrayLike | None = None,
        reference_temperature: float = 298.15,
        minimum_temperature: float = 1.0,
        maximum_temperature: float = 5000.0,
        thermodynamics_id: str | None = None,
    ):
        if not isinstance(schema, ChemicalSpeciesSchema):
            raise TypeError("schema must be a ChemicalSpeciesSchema.")
        coefficients = np.asarray(heat_capacity_volume, dtype=float)
        if coefficients.ndim == 1:
            coefficients = coefficients[:, None]
        reference_energy = np.asarray(reference_molar_internal_energy, dtype=float)
        reference_entropy = (
            np.zeros(schema.species_count, dtype=float)
            if reference_molar_entropy is None
            else np.asarray(reference_molar_entropy, dtype=float)
        )
        t_ref = float(reference_temperature)
        t_min = float(minimum_temperature)
        t_max = float(maximum_temperature)
        if (
            coefficients.ndim != 2
            or coefficients.shape[0] != schema.species_count
            or reference_energy.shape != (schema.species_count,)
            or reference_entropy.shape != (schema.species_count,)
            or np.any(~np.isfinite(coefficients))
            or np.any(~np.isfinite(reference_energy))
            or np.any(~np.isfinite(reference_entropy))
            or not 0.0 < t_min < t_ref < t_max
        ):
            raise ValueError("Polynomial thermodynamic inputs are invalid.")
        if any(
            not _positive_polynomial_on_interval(row, t_min, t_max)
            for row in coefficients
        ):
            raise ValueError("Molar heat capacity must remain positive over bounds.")
        generated = canonical_fingerprint(
            {
                "kind": "polynomial-species-thermodynamics",
                "schema": schema.schema_id,
                "capacity": array_tree_fingerprint(coefficients),
                "reference_energy": array_tree_fingerprint(reference_energy),
                "reference_entropy": array_tree_fingerprint(reference_entropy),
                "reference_temperature": t_ref,
                "bounds": [t_min, t_max],
            }
        )
        self.schema = schema
        self.heat_capacity_volume_coefficients = jnp.asarray(coefficients)
        self.reference_molar_internal_energy = jnp.asarray(reference_energy)
        self.reference_molar_entropy = jnp.asarray(reference_entropy)
        self.reference_temperature = t_ref
        self.minimum_temperature = t_min
        self.maximum_temperature = t_max
        self.thermodynamics_id = (
            generated if thermodynamics_id is None else str(thermodynamics_id)
        )
        if not self.thermodynamics_id:
            raise ValueError("thermodynamics_id must be nonempty.")

    def evaluate(self, temperature: ArrayLike, /) -> SpeciesThermodynamicEvaluation:
        value = jnp.asarray(temperature)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise ValueError("temperature must have inexact dtype.")
        valid = (
            jnp.isfinite(value)
            & (value >= self.minimum_temperature)
            & (value <= self.maximum_temperature)
        )
        safe = jnp.where(valid, value, self.reference_temperature)
        coefficient_count = self.heat_capacity_volume_coefficients.shape[1]
        powers = safe[..., None] ** jnp.arange(coefficient_count, dtype=safe.dtype)
        cv = contract("...k,sk->...s", powers, self.heat_capacity_volume_coefficients)
        integral_order = jnp.arange(1, coefficient_count + 1, dtype=safe.dtype)
        energy_integral = (
            safe[..., None] ** integral_order
            - jnp.asarray(self.reference_temperature, dtype=safe.dtype) ** integral_order
        ) / integral_order
        internal_energy = self.reference_molar_internal_energy + contract(
            "...k,sk->...s",
            energy_integral,
            self.heat_capacity_volume_coefficients,
        )
        logarithm = jnp.log(safe / self.reference_temperature)
        entropy_terms = []
        for order in range(coefficient_count):
            coefficient = self.heat_capacity_volume_coefficients[:, order]
            if order == 0:
                term = logarithm[..., None] * coefficient
            else:
                term = (
                    (safe[..., None] ** order - self.reference_temperature**order)
                    / float(order)
                    * coefficient
                )
            entropy_terms.append(term)
        entropy = self.reference_molar_entropy + sum(entropy_terms)
        gas_mask = self.schema.phase_mask(ChemicalPhaseKind.GAS).astype(safe.dtype)
        cp = cv + UNIVERSAL_GAS_CONSTANT * gas_mask
        enthalpy = internal_energy + (UNIVERSAL_GAS_CONSTANT * safe[..., None] * gas_mask)
        entropy = entropy + UNIVERSAL_GAS_CONSTANT * logarithm[..., None] * gas_mask
        gibbs = enthalpy - safe[..., None] * entropy
        successful = (
            valid
            & jnp.all(jnp.isfinite(cp), axis=-1)
            & jnp.all(jnp.isfinite(internal_energy), axis=-1)
            & jnp.all(cp > 0.0, axis=-1)
            & jnp.all(cv > 0.0, axis=-1)
        )
        margin = jnp.minimum(
            safe - self.minimum_temperature,
            self.maximum_temperature - safe,
        )
        interval = jnp.zeros(safe.shape + (self.schema.species_count,), dtype=jnp.int32)
        return SpeciesThermodynamicEvaluation(
            cp,
            cv,
            enthalpy,
            internal_energy,
            entropy,
            gibbs,
            interval,
            margin,
            successful,
            self.thermodynamics_id,
        )


class NASAPolynomialKind(StrEnum):
    NASA7 = "nasa7"
    NASA9 = "nasa9"


class NASASpeciesThermodynamicsPlan(AbstractSpeciesThermodynamicsPlan):
    """Piecewise NASA-7 or NASA-9 ideal-gas species thermodynamics."""

    polynomial_kind: NASAPolynomialKind = eqx.field(static=True)
    coefficients: Array
    lower_temperature: Array
    upper_temperature: Array
    interval_count: int = eqx.field(static=True)

    def __init__(
        self,
        schema: ChemicalSpeciesSchema,
        polynomial_kind: NASAPolynomialKind,
        coefficients: ArrayLike,
        lower_temperature: ArrayLike,
        upper_temperature: ArrayLike,
        /,
        *,
        thermodynamics_id: str | None = None,
    ):
        if not isinstance(schema, ChemicalSpeciesSchema):
            raise TypeError("schema must be a ChemicalSpeciesSchema.")
        if not isinstance(polynomial_kind, NASAPolynomialKind):
            raise TypeError("polynomial_kind must be NASAPolynomialKind.")
        values = np.asarray(coefficients, dtype=float)
        lower = np.asarray(lower_temperature, dtype=float)
        upper = np.asarray(upper_temperature, dtype=float)
        count = 7 if polynomial_kind is NASAPolynomialKind.NASA7 else 9
        if (
            values.ndim != 3
            or values.shape[0] != schema.species_count
            or values.shape[2] != count
            or lower.shape != values.shape[:2]
            or upper.shape != values.shape[:2]
            or np.any(~np.isfinite(values))
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(lower <= 0.0)
            or np.any(upper <= lower)
        ):
            raise ValueError("NASA polynomial arrays are invalid.")
        if values.shape[1] > 1 and not np.allclose(
            upper[:, :-1], lower[:, 1:], rtol=0.0, atol=0.0
        ):
            raise ValueError("NASA temperature intervals must be exactly contiguous.")
        minimum = float(np.min(lower[:, 0]))
        maximum = float(np.max(upper[:, -1]))
        generated = canonical_fingerprint(
            {
                "kind": polynomial_kind.value,
                "schema": schema.schema_id,
                "coefficients": array_tree_fingerprint(values),
                "lower": array_tree_fingerprint(lower),
                "upper": array_tree_fingerprint(upper),
            }
        )
        self.schema = schema
        self.polynomial_kind = polynomial_kind
        self.coefficients = jnp.asarray(values)
        self.lower_temperature = jnp.asarray(lower)
        self.upper_temperature = jnp.asarray(upper)
        self.interval_count = values.shape[1]
        self.minimum_temperature = minimum
        self.maximum_temperature = maximum
        self.thermodynamics_id = (
            generated if thermodynamics_id is None else str(thermodynamics_id)
        )
        if not self.thermodynamics_id:
            raise ValueError("thermodynamics_id must be nonempty.")

    def evaluate(self, temperature: ArrayLike, /) -> SpeciesThermodynamicEvaluation:
        value = jnp.asarray(temperature)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise ValueError("temperature must have inexact dtype.")
        expanded = value[..., None, None]
        lower = self.lower_temperature.astype(value.dtype)
        upper = self.upper_temperature.astype(value.dtype)
        interval_mask = (expanded >= lower) & (expanded <= upper)
        valid_species = jnp.any(interval_mask, axis=-1)
        interval = jnp.argmax(interval_mask, axis=-1).astype(jnp.int32)
        safe = jnp.where(
            jnp.all(valid_species, axis=-1),
            value,
            0.5 * (self.minimum_temperature + self.maximum_temperature),
        )
        gather_index = interval[..., None, None]
        coefficients = jnp.take_along_axis(
            jnp.broadcast_to(
                self.coefficients.astype(value.dtype),
                value.shape + self.coefficients.shape,
            ),
            gather_index,
            axis=-2,
        )[..., 0, :]
        if self.polynomial_kind is NASAPolynomialKind.NASA7:
            cp_r, h_rt, s_r = _nasa7(coefficients, safe[..., None])
        else:
            cp_r, h_rt, s_r = _nasa9(coefficients, safe[..., None])
        cp = UNIVERSAL_GAS_CONSTANT * cp_r
        enthalpy = UNIVERSAL_GAS_CONSTANT * safe[..., None] * h_rt
        entropy = UNIVERSAL_GAS_CONSTANT * s_r
        gas_mask = self.schema.phase_mask(ChemicalPhaseKind.GAS).astype(value.dtype)
        cv = cp - UNIVERSAL_GAS_CONSTANT * gas_mask
        internal_energy = enthalpy - (UNIVERSAL_GAS_CONSTANT * safe[..., None] * gas_mask)
        gibbs = enthalpy - safe[..., None] * entropy
        successful = (
            jnp.isfinite(value)
            & jnp.all(valid_species, axis=-1)
            & jnp.all(jnp.isfinite(cp), axis=-1)
            & jnp.all(cp > 0.0, axis=-1)
            & jnp.all(cv > 0.0, axis=-1)
        )
        species_lower = jnp.take_along_axis(lower, interval[..., None], axis=-1)[..., 0]
        species_upper = jnp.take_along_axis(upper, interval[..., None], axis=-1)[..., 0]
        margin = jnp.min(
            jnp.minimum(safe[..., None] - species_lower, species_upper - safe[..., None]),
            axis=-1,
        )
        return SpeciesThermodynamicEvaluation(
            cp,
            cv,
            enthalpy,
            internal_energy,
            entropy,
            gibbs,
            interval,
            margin,
            successful,
            self.thermodynamics_id,
        )


def _positive_polynomial_on_interval(
    coefficients: np.ndarray,
    lower: float,
    upper: float,
) -> bool:
    derivative = np.polynomial.polynomial.polyder(coefficients)
    roots = np.polynomial.polynomial.polyroots(derivative)
    real_roots = roots.real[
        (np.abs(roots.imag) <= 64.0 * np.finfo(float).eps)
        & (roots.real > lower)
        & (roots.real < upper)
    ]
    candidates = np.concatenate((np.asarray((lower, upper)), real_roots))
    values = np.polynomial.polynomial.polyval(candidates, coefficients)
    return bool(np.all(np.isfinite(values)) and np.all(values > 0.0))


def _nasa7(coefficients, temperature):
    a1, a2, a3, a4, a5, a6, a7 = jnp.moveaxis(coefficients, -1, 0)
    cp_r = a1 + temperature * (
        a2 + temperature * (a3 + temperature * (a4 + temperature * a5))
    )
    h_rt = (
        a1
        + a2 * temperature / 2.0
        + a3 * temperature**2 / 3.0
        + a4 * temperature**3 / 4.0
        + a5 * temperature**4 / 5.0
        + a6 / temperature
    )
    s_r = (
        a1 * jnp.log(temperature)
        + a2 * temperature
        + a3 * temperature**2 / 2.0
        + a4 * temperature**3 / 3.0
        + a5 * temperature**4 / 4.0
        + a7
    )
    return cp_r, h_rt, s_r


def _nasa9(coefficients, temperature):
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = jnp.moveaxis(coefficients, -1, 0)
    cp_r = (
        a1 / temperature**2
        + a2 / temperature
        + a3
        + a4 * temperature
        + a5 * temperature**2
        + a6 * temperature**3
        + a7 * temperature**4
    )
    h_rt = (
        -a1 / temperature**2
        + a2 * jnp.log(temperature) / temperature
        + a3
        + a4 * temperature / 2.0
        + a5 * temperature**2 / 3.0
        + a6 * temperature**3 / 4.0
        + a7 * temperature**4 / 5.0
        + a8 / temperature
    )
    s_r = (
        -a1 / (2.0 * temperature**2)
        - a2 / temperature
        + a3 * jnp.log(temperature)
        + a4 * temperature
        + a5 * temperature**2 / 2.0
        + a6 * temperature**3 / 3.0
        + a7 * temperature**4 / 4.0
        + a9
    )
    return cp_r, h_rt, s_r


__all__ = [
    "AbstractSpeciesThermodynamicsPlan",
    "NASAPolynomialKind",
    "NASASpeciesThermodynamicsPlan",
    "PolynomialSpeciesThermodynamicsPlan",
    "SpeciesThermodynamicEvaluation",
    "UNIVERSAL_GAS_CONSTANT",
]

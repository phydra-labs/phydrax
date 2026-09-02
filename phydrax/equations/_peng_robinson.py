#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_components import ChemicalComponentCatalog
from ._chemical_species import ChemicalSpeciesSchema
from ._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ._homogeneous_thermodynamics import (
    AbstractMolarHelmholtzTerm,
    HomogeneousHelmholtzPlan,
)


class PengRobinsonParameters(StrictModule, NonTrainableState):
    catalog: ChemicalComponentCatalog
    critical_temperature: Array
    critical_pressure: Array
    acentric_factor: Array
    binary_interaction: Array
    provenance: str = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        catalog: ChemicalComponentCatalog,
        critical_temperature: ArrayLike,
        critical_pressure: ArrayLike,
        acentric_factor: ArrayLike,
        binary_interaction: ArrayLike,
        /,
        *,
        provenance: str,
    ) -> None:
        if not isinstance(catalog, ChemicalComponentCatalog):
            raise TypeError("catalog must be ChemicalComponentCatalog.")
        temperature = np.asarray(critical_temperature, dtype=float)
        pressure = np.asarray(critical_pressure, dtype=float)
        acentric = np.asarray(acentric_factor, dtype=float)
        interaction = np.asarray(binary_interaction, dtype=float)
        count = catalog.component_count
        if temperature.shape != (count,) or pressure.shape != (count,):
            raise ValueError("Critical properties must have shape (component_count,).")
        if acentric.shape != (count,):
            raise ValueError("acentric_factor must have shape (component_count,).")
        if interaction.shape != (count, count):
            raise ValueError("binary_interaction must be a square component matrix.")
        if (
            np.any(~np.isfinite(temperature))
            or np.any(temperature <= 0.0)
            or np.any(~np.isfinite(pressure))
            or np.any(pressure <= 0.0)
            or np.any(~np.isfinite(acentric))
            or np.any(~np.isfinite(interaction))
        ):
            raise ValueError("Peng-Robinson parameters must be finite and physical.")
        if not np.array_equal(interaction, interaction.T):
            raise ValueError("binary_interaction must be exactly symmetric.")
        if not np.array_equal(np.diag(interaction), np.zeros((count,))):
            raise ValueError("binary_interaction diagonal must be exactly zero.")
        source = str(provenance)
        if not source:
            raise ValueError("provenance must be non-empty.")
        generated = canonical_fingerprint(
            {
                "kind": "peng-robinson-pr78-parameters",
                "catalog": catalog.catalog_id,
                "critical_temperature": array_tree_fingerprint(temperature),
                "critical_pressure": array_tree_fingerprint(pressure),
                "acentric_factor": array_tree_fingerprint(acentric),
                "binary_interaction": array_tree_fingerprint(interaction),
                "provenance": source,
            }
        )
        self.catalog = catalog
        self.critical_temperature = jnp.asarray(temperature)
        self.critical_pressure = jnp.asarray(pressure)
        self.acentric_factor = jnp.asarray(acentric)
        self.binary_interaction = jnp.asarray(interaction)
        self.provenance = source
        self.parameter_id = generated


class PengRobinsonResidualHelmholtzTerm(AbstractMolarHelmholtzTerm):
    """PR78 residual Helmholtz energy with quadratic-a and linear-b mixing."""

    parameters: PengRobinsonParameters

    def __init__(
        self,
        schema: ChemicalSpeciesSchema,
        parameters: PengRobinsonParameters,
        /,
    ) -> None:
        if not isinstance(schema, ChemicalSpeciesSchema):
            raise TypeError("schema must be ChemicalSpeciesSchema.")
        if not isinstance(parameters, PengRobinsonParameters):
            raise TypeError("parameters must be PengRobinsonParameters.")
        if schema.catalog.catalog_id != parameters.catalog.catalog_id:
            raise ValueError("Peng-Robinson parameters and schema catalogs must match.")
        if schema.species_count != schema.component_count or not np.array_equal(
            np.asarray(schema.species_component_indices),
            np.arange(schema.component_count),
        ):
            raise ValueError(
                "Peng-Robinson currently requires one species occurrence per component."
            )
        self.schema = schema
        self.parameters = parameters
        self.term_id = canonical_fingerprint(
            {
                "kind": "peng-robinson-pr78-residual",
                "schema": schema.schema_id,
                "parameters": parameters.parameter_id,
            }
        )

    def pure_parameters(self, temperature: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(temperature)
        critical_temperature = self.parameters.critical_temperature.astype(value.dtype)
        critical_pressure = self.parameters.critical_pressure.astype(value.dtype)
        acentric = self.parameters.acentric_factor.astype(value.dtype)
        low = 0.37464 + 1.54226 * acentric - 0.26992 * acentric**2
        high = (
            0.379642
            + 1.48503 * acentric
            - 0.164423 * acentric**2
            + 0.016666 * acentric**3
        )
        kappa = jnp.where(acentric > 0.491, high, low)
        alpha = (
            1.0 + kappa * (1.0 - jnp.sqrt(value[..., None] / critical_temperature))
        ) ** 2
        attractive = (
            0.4572355289
            * UNIVERSAL_GAS_CONSTANT**2
            * critical_temperature**2
            / critical_pressure
            * alpha
        )
        covolume = (
            0.0777960739
            * UNIVERSAL_GAS_CONSTANT
            * critical_temperature
            / critical_pressure
        )
        return attractive, covolume

    def mixture_parameters(
        self,
        temperature: ArrayLike,
        mole_fraction: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        composition = jnp.asarray(mole_fraction)
        attractive, covolume = self.pure_parameters(temperature)
        pair = jnp.sqrt(attractive[..., :, None] * attractive[..., None, :]) * (
            1.0 - self.parameters.binary_interaction.astype(composition.dtype)
        )
        mixture_attractive = contract(
            "...i,...ij,...j->...", composition, pair, composition, backend="jax"
        )
        mixture_covolume = contract(
            "...i,...i->...", composition, covolume, backend="jax"
        )
        return mixture_attractive, mixture_covolume

    def molar_helmholtz_energy(
        self,
        temperature: ArrayLike,
        molar_density: ArrayLike,
        mole_fraction: ArrayLike,
        /,
    ) -> Array:
        temperature_value = jnp.asarray(temperature)
        density = jnp.asarray(molar_density)
        composition = jnp.asarray(mole_fraction)
        attractive, covolume = self.mixture_parameters(temperature_value, composition)
        packing = covolume * density
        sqrt_two = jnp.sqrt(jnp.asarray(2.0, dtype=density.dtype))
        repulsive = -UNIVERSAL_GAS_CONSTANT * temperature_value * jnp.log1p(-packing)
        numerator = 1.0 + (1.0 + sqrt_two) * packing
        denominator = 1.0 + (1.0 - sqrt_two) * packing
        attractive_energy = (
            -attractive / (2.0 * sqrt_two * covolume) * jnp.log(numerator / denominator)
        )
        return repulsive + attractive_energy


class PengRobinsonRootSet(StrictModule):
    compressibility: Array
    molar_density: Array
    pressure_residual: Array
    molar_gibbs_energy: Array
    mechanical_derivative: Array
    valid: Array
    stable: Array
    multiplicity_margin: Array
    minimum_gibbs_index: Array
    successful: Array
    model_id: str = eqx.field(static=True)


def peng_robinson_roots(
    thermodynamics: HomogeneousHelmholtzPlan,
    temperature: ArrayLike,
    pressure: ArrayLike,
    mole_fraction: ArrayLike,
    /,
) -> PengRobinsonRootSet:
    """Return all three fixed-capacity PR compressibility root slots."""
    if not isinstance(thermodynamics.residual, PengRobinsonResidualHelmholtzTerm):
        raise TypeError("thermodynamics must contain a Peng-Robinson residual term.")
    temperature_value = jnp.asarray(temperature)
    pressure_value = jnp.asarray(pressure)
    composition = jnp.asarray(mole_fraction)
    if temperature_value.shape != () or pressure_value.shape != ():
        raise ValueError("Peng-Robinson root enumeration currently accepts one state.")
    if composition.shape != (thermodynamics.schema.component_count,):
        raise ValueError("mole_fraction must have shape (component_count,).")
    attractive, covolume = thermodynamics.residual.mixture_parameters(
        temperature_value, composition
    )
    reduced_attraction = (
        attractive * pressure_value / (UNIVERSAL_GAS_CONSTANT**2 * temperature_value**2)
    )
    reduced_covolume = (
        covolume * pressure_value / (UNIVERSAL_GAS_CONSTANT * temperature_value)
    )
    coefficients = jnp.asarray(
        (
            -(1.0 - reduced_covolume),
            reduced_attraction - 3.0 * reduced_covolume**2 - 2.0 * reduced_covolume,
            -(
                reduced_attraction * reduced_covolume
                - reduced_covolume**2
                - reduced_covolume**3
            ),
        )
    )
    roots, discriminant = _monic_cubic_real_roots(*coefficients)
    valid = jnp.isfinite(roots) & (roots > reduced_covolume) & (roots > 0.0)
    safe_roots = jnp.where(valid, roots, 1.0)
    density = pressure_value / (safe_roots * UNIVERSAL_GAS_CONSTANT * temperature_value)
    state = jax.vmap(
        lambda density_value: thermodynamics.evaluate(
            temperature_value, density_value, composition
        )
    )(density)
    pressure_residual = state.pressure - pressure_value
    mechanical = state.pressure_molar_density_derivative
    stable = valid & state.evidence.successful & (mechanical > 0.0)
    gibbs = jnp.where(stable, state.molar_gibbs_energy, jnp.inf)
    minimum_index = jnp.argmin(gibbs).astype(jnp.int32)
    sorted_finite = jnp.where(valid, roots, jnp.inf)
    separations = jnp.abs(sorted_finite[1:] - sorted_finite[:-1])
    multiplicity_margin = jnp.minimum(
        jnp.min(separations, initial=jnp.inf), jnp.sqrt(jnp.abs(discriminant))
    )
    successful = (
        jnp.isfinite(temperature_value)
        & (temperature_value > 0.0)
        & jnp.isfinite(pressure_value)
        & (pressure_value > 0.0)
        & jnp.any(stable)
    )
    return PengRobinsonRootSet(
        roots,
        density,
        pressure_residual,
        gibbs,
        mechanical,
        valid,
        stable,
        multiplicity_margin,
        minimum_index,
        successful,
        thermodynamics.model_id,
    )


def _monic_cubic_real_roots(a, b, c):
    p = b - a**2 / 3.0
    q = 2.0 * a**3 / 27.0 - a * b / 3.0 + c
    discriminant = (0.5 * q) ** 2 + (p / 3.0) ** 3
    sqrt_discriminant = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    one = (
        jnp.cbrt(-0.5 * q + sqrt_discriminant)
        + jnp.cbrt(-0.5 * q - sqrt_discriminant)
        - a / 3.0
    )
    safe_p = jnp.minimum(p, -jnp.finfo(jnp.result_type(a, b, c)).eps)
    radius = 2.0 * jnp.sqrt(-safe_p / 3.0)
    cosine = (3.0 * q / (2.0 * safe_p)) * jnp.sqrt(-3.0 / safe_p)
    angle = jnp.arccos(jnp.clip(cosine, -1.0, 1.0)) / 3.0
    offsets = 2.0 * jnp.pi * jnp.arange(3, dtype=angle.dtype) / 3.0
    three = radius * jnp.cos(angle - offsets) - a / 3.0
    roots = jnp.where(
        discriminant <= 0.0,
        three,
        jnp.asarray((one, jnp.nan, jnp.nan), dtype=angle.dtype),
    )
    roots = jnp.sort(jnp.where(jnp.isfinite(roots), roots, jnp.inf))
    roots = jnp.where(jnp.isfinite(roots), roots, jnp.nan)
    return roots, discriminant


__all__ = [
    "PengRobinsonParameters",
    "PengRobinsonResidualHelmholtzTerm",
    "PengRobinsonRootSet",
    "peng_robinson_roots",
]

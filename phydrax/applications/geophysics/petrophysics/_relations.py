#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule


class PetrophysicalPrediction(StrictModule):
    mean: Array
    standard_deviation: Array


class SurfaceConductionConductivity(StrictModule):
    water_conductivity_S_m: Array
    cementation_exponent: Array
    saturation_exponent: Array
    surface_conductivity_S_m: Array
    surface_exponent: Array
    discrepancy_standard_deviation_log: Array

    def __init__(
        self,
        water_conductivity_S_m: ArrayLike,
        cementation_exponent: ArrayLike,
        saturation_exponent: ArrayLike,
        surface_conductivity_S_m: ArrayLike,
        surface_exponent: ArrayLike,
        discrepancy_standard_deviation_log: ArrayLike,
        /,
    ):
        values = jnp.broadcast_arrays(
            *(
                jnp.asarray(value)
                for value in (
                    water_conductivity_S_m,
                    cementation_exponent,
                    saturation_exponent,
                    surface_conductivity_S_m,
                    surface_exponent,
                    discrepancy_standard_deviation_log,
                )
            )
        )
        invalid = any(jnp.any(~jnp.isfinite(value)) for value in values)
        invalid = (
            invalid
            | any(jnp.any(value <= 0) for value in values[:5])
            | jnp.any(values[5] < 0)
        )
        self.water_conductivity_S_m = eqx.error_if(
            values[0],
            invalid,
            "Surface-conduction calibration must be finite and physical.",
        )
        (
            self.cementation_exponent,
            self.saturation_exponent,
            self.surface_conductivity_S_m,
            self.surface_exponent,
            self.discrepancy_standard_deviation_log,
        ) = values[1:]

    def predict(
        self, porosity: ArrayLike, saturation: ArrayLike, clay_fraction: ArrayLike, /
    ) -> PetrophysicalPrediction:
        porosity_, saturation_, clay = jnp.broadcast_arrays(
            jnp.asarray(porosity), jnp.asarray(saturation), jnp.asarray(clay_fraction)
        )
        porosity_ = eqx.error_if(
            porosity_,
            jnp.any(~jnp.isfinite(porosity_))
            | jnp.any((porosity_ <= 0) | (porosity_ > 1))
            | jnp.any(~jnp.isfinite(saturation_))
            | jnp.any((saturation_ <= 0) | (saturation_ > 1))
            | jnp.any(~jnp.isfinite(clay))
            | jnp.any((clay < 0) | (clay > 1)),
            "Surface-conduction inputs must lie in their physical fractions.",
        )
        bulk = (
            self.water_conductivity_S_m
            * porosity_**self.cementation_exponent
            * saturation_**self.saturation_exponent
            + self.surface_conductivity_S_m * clay**self.surface_exponent
        )
        return PetrophysicalPrediction(
            bulk, bulk * self.discrepancy_standard_deviation_log
        )


class CRIMPermittivity(StrictModule):
    solid_relative_permittivity: Array
    water_relative_permittivity: Array
    air_relative_permittivity: Array
    discrepancy_standard_deviation: Array

    def __init__(
        self,
        solid_relative_permittivity: ArrayLike,
        water_relative_permittivity: ArrayLike,
        air_relative_permittivity: ArrayLike = 1.0,
        discrepancy_standard_deviation: ArrayLike = 0.0,
        /,
    ):
        values = jnp.broadcast_arrays(
            jnp.asarray(solid_relative_permittivity),
            jnp.asarray(water_relative_permittivity),
            jnp.asarray(air_relative_permittivity),
            jnp.asarray(discrepancy_standard_deviation),
        )
        invalid = any(jnp.any(~jnp.isfinite(value)) for value in values)
        invalid = (
            invalid
            | any(jnp.any(value <= 0) for value in values[:3])
            | jnp.any(values[3] < 0)
        )
        self.solid_relative_permittivity = eqx.error_if(
            values[0], invalid, "CRIM calibration must be finite and physical."
        )
        (
            self.water_relative_permittivity,
            self.air_relative_permittivity,
            self.discrepancy_standard_deviation,
        ) = values[1:]

    def predict(
        self, porosity: ArrayLike, water_saturation: ArrayLike, /
    ) -> PetrophysicalPrediction:
        porosity, saturation = jnp.broadcast_arrays(
            jnp.asarray(porosity), jnp.asarray(water_saturation)
        )
        porosity = eqx.error_if(
            porosity,
            jnp.any(~jnp.isfinite(porosity))
            | jnp.any((porosity < 0) | (porosity > 1))
            | jnp.any(~jnp.isfinite(saturation))
            | jnp.any((saturation < 0) | (saturation > 1)),
            "CRIM porosity and water saturation must be finite fractions.",
        )
        root = (
            (1 - porosity) * jnp.sqrt(self.solid_relative_permittivity)
            + porosity * saturation * jnp.sqrt(self.water_relative_permittivity)
            + porosity * (1 - saturation) * jnp.sqrt(self.air_relative_permittivity)
        )
        return PetrophysicalPrediction(root**2, self.discrepancy_standard_deviation)


class GassmannFluidSubstitution(StrictModule):
    mineral_bulk_modulus_Pa: Array
    dry_bulk_modulus_Pa: Array
    dry_shear_modulus_Pa: Array
    mineral_density_kg_m3: Array
    discrepancy_standard_deviation_Pa: Array

    def __init__(
        self,
        mineral_bulk_modulus_Pa: ArrayLike,
        dry_bulk_modulus_Pa: ArrayLike,
        dry_shear_modulus_Pa: ArrayLike,
        mineral_density_kg_m3: ArrayLike,
        discrepancy_standard_deviation_Pa: ArrayLike = 0.0,
        /,
    ):
        values = jnp.broadcast_arrays(
            jnp.asarray(mineral_bulk_modulus_Pa),
            jnp.asarray(dry_bulk_modulus_Pa),
            jnp.asarray(dry_shear_modulus_Pa),
            jnp.asarray(mineral_density_kg_m3),
            jnp.asarray(discrepancy_standard_deviation_Pa),
        )
        invalid = any(jnp.any(~jnp.isfinite(value)) for value in values)
        invalid = (
            invalid
            | jnp.any(values[0] <= 0)
            | jnp.any(values[1] <= 0)
            | jnp.any(values[1] >= values[0])
            | jnp.any(values[2] <= 0)
            | jnp.any(values[3] <= 0)
            | jnp.any(values[4] < 0)
        )
        self.mineral_bulk_modulus_Pa = eqx.error_if(
            values[0], invalid, "Gassmann calibration must be finite and physical."
        )
        (
            self.dry_bulk_modulus_Pa,
            self.dry_shear_modulus_Pa,
            self.mineral_density_kg_m3,
            self.discrepancy_standard_deviation_Pa,
        ) = values[1:]

    def predict(
        self,
        porosity: ArrayLike,
        fluid_bulk_modulus_Pa: ArrayLike,
        fluid_density_kg_m3: ArrayLike,
        /,
    ) -> tuple[PetrophysicalPrediction, Array, Array, Array]:
        porosity, fluid_bulk, fluid_density = jnp.broadcast_arrays(
            jnp.asarray(porosity),
            jnp.asarray(fluid_bulk_modulus_Pa),
            jnp.asarray(fluid_density_kg_m3),
        )
        porosity = eqx.error_if(
            porosity,
            jnp.any(~jnp.isfinite(porosity))
            | jnp.any((porosity <= 0) | (porosity >= 1))
            | jnp.any(~jnp.isfinite(fluid_bulk))
            | jnp.any(fluid_bulk <= 0)
            | jnp.any(~jnp.isfinite(fluid_density))
            | jnp.any(fluid_density <= 0),
            "Gassmann porosity and fluid properties must be finite and physical.",
        )
        denominator = (
            porosity / fluid_bulk
            + (1 - porosity) / self.mineral_bulk_modulus_Pa
            - self.dry_bulk_modulus_Pa / self.mineral_bulk_modulus_Pa**2
        )
        denominator = eqx.error_if(
            denominator,
            jnp.any(~jnp.isfinite(denominator)) | jnp.any(denominator <= 0),
            "Gassmann fluid substitution has a nonpositive bulk denominator.",
        )
        saturated_bulk = (
            self.dry_bulk_modulus_Pa
            + (1 - self.dry_bulk_modulus_Pa / self.mineral_bulk_modulus_Pa) ** 2
            / denominator
        )
        density = (1 - porosity) * self.mineral_density_kg_m3 + porosity * fluid_density
        p_velocity = jnp.sqrt(
            (saturated_bulk + 4 * self.dry_shear_modulus_Pa / 3) / density
        )
        s_velocity = jnp.sqrt(self.dry_shear_modulus_Pa / density)
        return (
            PetrophysicalPrediction(
                saturated_bulk, self.discrepancy_standard_deviation_Pa
            ),
            self.dry_shear_modulus_Pa,
            density,
            jnp.stack((p_velocity, s_velocity), axis=-1),
        )


class KozenyCarmanPermeability(StrictModule):
    reference_permeability_m2: Array
    reference_porosity: Array
    exponent: Array
    discrepancy_standard_deviation_log: Array

    def __init__(
        self,
        reference_permeability_m2: ArrayLike,
        reference_porosity: ArrayLike,
        exponent: ArrayLike = 2.0,
        discrepancy_standard_deviation_log: ArrayLike = 0.0,
        /,
    ):
        values = jnp.broadcast_arrays(
            jnp.asarray(reference_permeability_m2),
            jnp.asarray(reference_porosity),
            jnp.asarray(exponent),
            jnp.asarray(discrepancy_standard_deviation_log),
        )
        invalid = any(jnp.any(~jnp.isfinite(value)) for value in values)
        invalid = (
            invalid
            | jnp.any(values[0] <= 0)
            | jnp.any((values[1] <= 0) | (values[1] >= 1))
            | jnp.any(values[2] <= 0)
            | jnp.any(values[3] < 0)
        )
        self.reference_permeability_m2 = eqx.error_if(
            values[0], invalid, "Kozeny-Carman calibration must be finite and physical."
        )
        (
            self.reference_porosity,
            self.exponent,
            self.discrepancy_standard_deviation_log,
        ) = values[1:]

    def predict(self, porosity: ArrayLike, /) -> PetrophysicalPrediction:
        value = jnp.asarray(porosity)
        value = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)) | jnp.any((value <= 0) | (value >= 1)),
            "Kozeny-Carman porosity must be a finite open-unit fraction.",
        )
        ratio = (value / self.reference_porosity) ** 3 * (
            (1 - self.reference_porosity) / (1 - value)
        ) ** self.exponent
        mean = self.reference_permeability_m2 * ratio
        return PetrophysicalPrediction(
            mean, mean * self.discrepancy_standard_deviation_log
        )


__all__ = [
    "CRIMPermittivity",
    "GassmannFluidSubstitution",
    "KozenyCarmanPermeability",
    "PetrophysicalPrediction",
    "SurfaceConductionConductivity",
]

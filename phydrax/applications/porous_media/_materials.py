#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""SI-qualified liquid and porous-skeleton properties without numerical floors."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...discretization.finite_volume._hybrid_diffusion import _positive_tensor, _tensor
from ...units import (
    convert_value,
    derived_unit,
    KELVIN,
    KILOGRAM,
    METER,
    PASCAL,
    SECOND,
    UnitDefinition,
)


SQUARE_METER = derived_unit("m2", ((METER, 2),))
DENSITY_UNIT = derived_unit("kg/m3", ((KILOGRAM, 1), (METER, -3)))
VISCOSITY_UNIT = derived_unit("Pa.s", ((PASCAL, 1), (SECOND, 1)))
INVERSE_PASCAL = derived_unit("1/Pa", ((PASCAL, -1),))
INVERSE_KELVIN = derived_unit("1/K", ((KELVIN, -1),))


def _finite(value, name, *, positive=False, nonnegative=False):
    result = jnp.asarray(value)
    result = result.astype(jnp.result_type(result, 1.0))
    invalid = ~jnp.all(jnp.isfinite(result))
    if positive:
        invalid = invalid | jnp.any(result <= 0)
    if nonnegative:
        invalid = invalid | jnp.any(result < 0)
    return eqx.error_if(
        result, invalid, f"{name} must be finite and physically admissible."
    )


class PorousMaterial(StrictModule):
    """Intrinsic SPD permeability, porosity, and explicit liquid property laws.

    rho = rho_ref exp(c_rho*(p-p_ref) - alpha_T*(T-T_ref));
    mu = mu_ref exp(-beta_mu*(T-T_ref));
    phi = phi_ref exp(c_phi*(p-p_ref)). Zero coefficients mean exactly constant
    properties, not a hidden storage regularization. Positive absolute temperature
    and 0 < phi < 1 define the constitutive domain. Pressure is gauge pressure.
    """

    porosity: Array
    permeability_m2: Array
    density_kg_m3: Array
    viscosity_Pa_s: Array
    fluid_compressibility_Pa_inverse: Array
    pore_compressibility_Pa_inverse: Array
    thermal_expansion_K_inverse: Array
    viscosity_temperature_K_inverse: Array
    reference_pressure_Pa: Array
    reference_temperature_K: Array

    def __init__(
        self,
        porosity,
        permeability_m2,
        /,
        *,
        density_kg_m3=1000.0,
        viscosity_Pa_s=1.0e-3,
        fluid_compressibility_Pa_inverse=0.0,
        pore_compressibility_Pa_inverse=0.0,
        thermal_expansion_K_inverse=0.0,
        viscosity_temperature_K_inverse=0.0,
        reference_pressure_Pa=0.0,
        reference_temperature_K=293.15,
        permeability_unit: UnitDefinition = SQUARE_METER,
        density_unit: UnitDefinition = DENSITY_UNIT,
        viscosity_unit: UnitDefinition = VISCOSITY_UNIT,
    ):
        phi = _finite(porosity, "porosity", positive=True)
        self.porosity = eqx.error_if(
            phi, jnp.any(phi >= 1), "porosity must be less than one."
        )
        permeability = convert_value(
            permeability_m2, source=permeability_unit, target=SQUARE_METER
        )
        permeability = _finite(permeability, "permeability")
        if permeability.ndim in (0, 1):
            permeability = eqx.error_if(
                permeability,
                jnp.any(permeability <= 0),
                "Intrinsic permeability must be positive.",
            )
        elif permeability.ndim in (2, 3) and permeability.shape[-2:] == (3, 3):
            _positive_tensor(
                _tensor(
                    permeability, 1 if permeability.ndim == 2 else permeability.shape[0]
                )
            )
        else:
            raise ValueError(
                "permeability must be scalar, cell scalar, or full 3D SPD tensor."
            )
        self.permeability_m2 = permeability
        self.density_kg_m3 = _finite(
            convert_value(density_kg_m3, source=density_unit, target=DENSITY_UNIT),
            "density",
            positive=True,
        )
        self.viscosity_Pa_s = _finite(
            convert_value(viscosity_Pa_s, source=viscosity_unit, target=VISCOSITY_UNIT),
            "viscosity",
            positive=True,
        )
        self.fluid_compressibility_Pa_inverse = _finite(
            fluid_compressibility_Pa_inverse, "fluid compressibility", nonnegative=True
        )
        self.pore_compressibility_Pa_inverse = _finite(
            pore_compressibility_Pa_inverse, "pore compressibility", nonnegative=True
        )
        self.thermal_expansion_K_inverse = _finite(
            thermal_expansion_K_inverse, "thermal expansion", nonnegative=True
        )
        self.viscosity_temperature_K_inverse = _finite(
            viscosity_temperature_K_inverse,
            "viscosity temperature coefficient",
            nonnegative=True,
        )
        self.reference_pressure_Pa = _finite(reference_pressure_Pa, "reference pressure")
        self.reference_temperature_K = _finite(
            reference_temperature_K, "reference temperature", positive=True
        )

    def density(self, pressure_Pa, temperature_K):
        return self.density_kg_m3 * jnp.exp(
            self.fluid_compressibility_Pa_inverse
            * (pressure_Pa - self.reference_pressure_Pa)
            - self.thermal_expansion_K_inverse
            * (temperature_K - self.reference_temperature_K)
        )

    def viscosity(self, temperature_K):
        return self.viscosity_Pa_s * jnp.exp(
            -self.viscosity_temperature_K_inverse
            * (temperature_K - self.reference_temperature_K)
        )

    def pore_fraction(self, pressure_Pa):
        return self.porosity * jnp.exp(
            self.pore_compressibility_Pa_inverse
            * (pressure_Pa - self.reference_pressure_Pa)
        )

    def admissible(self, pressure_Pa, temperature_K):
        phi = self.pore_fraction(pressure_Pa)
        rho, mu = self.density(pressure_Pa, temperature_K), self.viscosity(temperature_K)
        return jnp.all(
            jnp.isfinite(pressure_Pa)
            & jnp.isfinite(temperature_K)
            & (temperature_K > 0)
            & jnp.isfinite(phi)
            & (phi > 0)
            & (phi < 1)
            & jnp.isfinite(rho)
            & (rho > 0)
            & jnp.isfinite(mu)
            & (mu > 0)
        )


__all__ = ["PorousMaterial"]

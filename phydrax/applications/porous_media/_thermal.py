#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Local thermal equilibrium of a fixed porous skeleton and one liquid phase."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...discretization.finite_volume._hybrid_diffusion import _positive_tensor, _tensor
from ...units import (
    convert_value,
    derived_unit,
    JOULE,
    KELVIN,
    METER,
    SECOND,
    UnitDefinition,
)
from ._materials import _finite


CONDUCTIVITY_UNIT = derived_unit(
    "W/(m.K)", ((JOULE, 1), (SECOND, -1), (METER, -1), (KELVIN, -1))
)


class PorousThermalMaterial(StrictModule):
    """Sensible-energy constitutive law, with no latent heat, vapor or freezing.

    Effective conductivity tensors of the dry and saturated medium interpolate
    linearly in liquid saturation. The fixed skeleton contributes its reference
    solid volume times ``solid_heat_capacity_J_m3_K``. Liquid enthalpy is
    cp*(T-T_ref), and liquid sensible inventory is water_mass*enthalpy. This is
    the local-thermal-equilibrium porous heat model, not a compressible total-
    energy equation with kinetic energy, elastic work or viscous dissipation.
    """

    dry_conductivity_W_m_K: Array
    saturated_conductivity_W_m_K: Array
    solid_heat_capacity_J_m3_K: Array
    liquid_heat_capacity_J_kg_K: Array
    reference_temperature_K: Array

    def __init__(
        self,
        conductivity_W_m_K,
        /,
        *,
        dry_conductivity_W_m_K=None,
        solid_heat_capacity_J_m3_K=2.0e6,
        liquid_heat_capacity_J_kg_K=4180.0,
        reference_temperature_K=273.15,
        conductivity_unit: UnitDefinition = CONDUCTIVITY_UNIT,
    ):
        saturated = convert_value(
            conductivity_W_m_K, source=conductivity_unit, target=CONDUCTIVITY_UNIT
        )
        dry = (
            saturated
            if dry_conductivity_W_m_K is None
            else convert_value(
                dry_conductivity_W_m_K, source=conductivity_unit, target=CONDUCTIVITY_UNIT
            )
        )
        self.saturated_conductivity_W_m_K = _finite(saturated, "saturated conductivity")
        self.dry_conductivity_W_m_K = _finite(dry, "dry conductivity")
        for value in (dry, saturated):
            count = value.shape[0] if value.ndim in (1, 3) else 1
            _positive_tensor(_tensor(value, count))
        self.solid_heat_capacity_J_m3_K = _finite(
            solid_heat_capacity_J_m3_K, "solid volumetric heat capacity", positive=True
        )
        self.liquid_heat_capacity_J_kg_K = _finite(
            liquid_heat_capacity_J_kg_K, "liquid specific heat capacity", positive=True
        )
        self.reference_temperature_K = _finite(
            reference_temperature_K, "thermal reference temperature", positive=True
        )
        if (
            self.liquid_heat_capacity_J_kg_K.shape != ()
            or self.reference_temperature_K.shape != ()
        ):
            raise ValueError(
                "One liquid requires scalar specific heat and a common enthalpy reference."
            )

    def conductivity(self, saturation):
        count = saturation.size
        dry = _tensor(self.dry_conductivity_W_m_K, count)
        saturated = _tensor(self.saturated_conductivity_W_m_K, count)
        return dry + saturation[:, None, None] * (saturated - dry)

    def enthalpy(self, temperature_K):
        return self.liquid_heat_capacity_J_kg_K * (
            jnp.asarray(temperature_K) - self.reference_temperature_K
        )

    def energy(self, water_state, discretization, porous_material):
        skeleton_capacity = (
            (1 - porous_material.porosity)
            * discretization.cell_volumes
            * self.solid_heat_capacity_J_m3_K
        )
        return skeleton_capacity * (
            water_state.temperature_K - self.reference_temperature_K
        ) + water_state.water_mass_kg * self.enthalpy(water_state.temperature_K)


class PorousHeatFluxes(StrictModule):
    """Conduction and liquid-mass-upwind enthalpy rates, in W, owner oriented."""

    conductive_face_rates_W: Array
    advective_face_rates_W: Array
    total_face_rates_W: Array
    local_conductive_rates_W: Array
    local_advective_rates_W: Array


__all__ = ["PorousHeatFluxes", "PorousThermalMaterial"]

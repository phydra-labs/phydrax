#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Communicating-volume closure and plenum spring in the current geometry."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ._free_volume import (
    FreeVolumeComponents,
    FreeVolumeModel,
    plenum_and_void_volumes,
)
from ._geometry import FuelRodGeometry
from ._numerical_views import FuelNumericalSpecification, NumericalFreeVolumeModel
from ._specification import FuelRodSpecification
from ._state import FuelRodFields
from ._zones import zone_average


class PlenumResponse(StrictModule):
    """Current free-gas volumes, spring force [N], and zone-mean kinematics.

    Radii [m] and axial strains are weighted means over each axial zone.
    """

    components: FreeVolumeComponents
    spring_force: Array
    fuel_surface_radius: Array
    cladding_inner_radius: Array
    cladding_outer_radius: Array
    axial_strain_fuel: Array
    axial_strain_cladding: Array


def plenum_response(
    specification: FuelRodSpecification | FuelNumericalSpecification,
    geometry: FuelRodGeometry,
    free_volume: FreeVolumeModel | NumericalFreeVolumeModel,
    fields: FuelRodFields,
    fuel_mean_temperature: Array,
    relocation: Array,
    densification_loss: Array,
    connected_porosity: Array,
    plenum_temperature: Array,
    mechanical_relocation: Array,
    fuel_thermal_stretch: Array,
    /,
) -> PlenumResponse:
    """Free-gas volumes and the plenum spring force of the current rod state.

    Zone arrays (``relocation``, ``densification_loss``, ``connected_porosity``,
    ``mechanical_relocation``, ``fuel_thermal_stretch``, ``fuel_mean_temperature``)
    have one entry per axial zone. The gap volume and gas temperature are
    resolved per fuel-interface site.
    """

    zone_count = geometry.layer_lengths.shape[0]
    fuel_interface = geometry.fuel_interface
    cladding_interface = geometry.cladding_interface
    cladding_outer_surface = geometry.cladding_outer
    fuel_interface_displacement = fuel_interface.values(fields.displacement_fuel)
    cladding_interface_displacement = cladding_interface.values(
        fields.displacement_cladding
    )
    cladding_outer_displacement = cladding_outer_surface.values(
        fields.displacement_cladding
    )
    fuel_interface_radius = (
        fuel_interface.reference_radius + fuel_interface_displacement[:, 0]
    )
    cladding_interface_radius = (
        cladding_interface.reference_radius + cladding_interface_displacement[:, 0]
    )
    mean_fuel_radius = zone_average(
        fuel_interface_radius, fuel_interface.weight, fuel_interface.zone, zone_count
    )
    mean_cladding_inner_radius = zone_average(
        cladding_interface_radius,
        cladding_interface.weight,
        cladding_interface.zone,
        zone_count,
    )
    mean_cladding_outer_radius = zone_average(
        cladding_outer_surface.reference_radius + cladding_outer_displacement[:, 0],
        cladding_outer_surface.weight,
        cladding_outer_surface.zone,
        zone_count,
    )
    if geometry.layered:
        fuel_axial_strain = fields.axial_strain_fuel
        cladding_axial_strain = fields.axial_strain_cladding
    else:
        fuel_displacement_gradient = geometry.fuel.gradients(fields.displacement_fuel)
        cladding_displacement_gradient = geometry.cladding.gradients(
            fields.displacement_cladding
        )
        fuel_axial_strain = zone_average(
            fuel_displacement_gradient[:, 1, 1],
            geometry.fuel.site_weight,
            geometry.fuel.site_zone,
            zone_count,
        )
        cladding_axial_strain = zone_average(
            cladding_displacement_gradient[:, 1, 1],
            geometry.cladding.site_weight,
            geometry.cladding.site_zone,
            zone_count,
        )
    fuel_elongation = jnp.sum(geometry.layer_lengths * fuel_axial_strain)
    cladding_elongation = jnp.sum(geometry.layer_lengths * cladding_axial_strain)
    spring_force = (
        specification.rod.spring_preload
        + specification.rod.spring_stiffness * (fuel_elongation - cladding_elongation)
    )
    plenum_length = (
        specification.rod.plenum_length * (1 + cladding_axial_strain[-1])
        + cladding_elongation
        - fuel_elongation
    )
    gap_gas_temperature = (
        fuel_interface.values(fields.temperature_fuel)
        + cladding_interface.values(fields.temperature_cladding)
    ) / 2
    plenum_volume, void_volume = plenum_and_void_volumes(
        free_volume,
        layer_lengths=geometry.layer_lengths,
        plenum_length=plenum_length,
        plenum_bore_radius=mean_cladding_inner_radius[-1],
        relocation_strain=relocation,
        densification_volume_fraction=densification_loss,
    )
    connected_volume = (
        jnp.pi
        * specification.pellet.radius**2
        * geometry.layer_lengths
        * connected_porosity
    )
    # Dish/chamfer voids expand with the material, not with the annular gap.
    fabricated_dishes = (
        free_volume.dish_chamfer_volume_per_length * geometry.layer_lengths
    )
    dish_correction = fabricated_dishes * (fuel_thermal_stretch**3 - 1)
    # The thermal gap excludes the mechanical part of the relocation strain and
    # never extends past the cladding bore.
    thermal_fuel_radius = jnp.minimum(
        fuel_interface_radius
        + specification.pellet.radius
        * (relocation - mechanical_relocation)[fuel_interface.zone],
        cladding_interface_radius,
    )
    axial_stretch = (
        1 + fuel_axial_strain[fuel_interface.zone]
        if geometry.layered
        else 1 + fuel_interface.axial_gradient(fields.displacement_fuel)[:, 1]
    )
    # Trace weights carry 2 pi R dz; dividing it out leaves the annulus area
    # times the current axial length of each site.
    gap_volume = (
        jnp.pi
        * (cladding_interface_radius**2 - thermal_fuel_radius**2)
        * axial_stretch
        * fuel_interface.weight
        / (2 * jnp.pi * fuel_interface.reference_radius)
    )
    return PlenumResponse(
        components=FreeVolumeComponents(
            plenum_volume=plenum_volume,
            plenum_temperature=plenum_temperature,
            gap_volume=gap_volume,
            gap_temperature=gap_gas_temperature,
            void_volume=void_volume + connected_volume + dish_correction,
            void_temperature=fuel_mean_temperature,
        ),
        spring_force=spring_force,
        fuel_surface_radius=mean_fuel_radius,
        cladding_inner_radius=mean_cladding_inner_radius,
        cladding_outer_radius=mean_cladding_outer_radius,
        axial_strain_fuel=fuel_axial_strain,
        axial_strain_cladding=cladding_axial_strain,
    )

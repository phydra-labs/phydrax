"""Total-Lagrangian rod residual forms on the prepared FE routes.

Residual rows use the ``FuelRodFields`` layout: each row block sits in the slot
of the unknown it determines, so the contact-gap row pairs with the contact
pressure in the complementarity map. The row scales below only condition the
solver; they are not physical tolerances.
"""

from __future__ import annotations

from typing import assert_never, Final, final, NamedTuple, Protocol, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ...applications.contact import (
    axisymmetric_interface_geometry,
    AxisymmetricInterfaceGeometry,
    coulomb_interface_response,
)
from ...nuclear import (
    advance_exposure,
    ExposureState,
    FISSIONING_NUCLIDES,
    HeavyMetalReference,
)
from ...operators.mechanics import (
    axisymmetric_deformation_gradient,
    finite_strain_kinematics,
)
from ._bundles import FuelModelBundle
from ._cladding_properties import ZIRCALOY_DENSITY
from ._constants import (
    AVOGADRO_PER_MOLE,
    JOULE_PER_MEGAWATT_DAY,
    MOLAR_GAS_CONSTANT_J_PER_MOL_K,
)
from ._coolant import PrescribedWallTemperature, SingleChannelCoolant
from ._coupled_coolant import CoupledChannelCoolant
from ._free_volume import FreeVolumeModel, gas_pressure_residual
from ._fuel_constitutive import (
    cladding_response,
    CladdingConstitutiveResponse,
    fuel_response,
    FuelConstitutiveResponse,
)
from ._fuel_properties import UO2_THEORETICAL_DENSITY
from ._gap import GapConductance
from ._gas_block import advance_fuel_gas, FuelGasBlockResult
from ._gas_state import FissionGasSiteState, FissionGasSpeciesState
from ._geometry import FuelBodyGeometry, FuelRodGeometry
from ._helium import HeliumGasSiteState
from ._identity import FUEL_GAS_SPECIES
from ._inventory import FuelEnergyLedger, FuelGasLedger
from ._irradiation import IrradiationConditions
from ._layout import (
    CLADDING_SITE_SOURCE_SLOTS,
    FUEL_SITE_SOURCE_SLOTS,
    ROD_SOURCE_SLOTS,
    STATICALLY_UNSOURCED_SLOTS,
)
from ._numerical_views import NumericalFreeVolumeModel
from ._plenum import plenum_response, PlenumResponse
from ._radial_power import RadialPowerState
from ._site_state import CladdingSiteState, FuelSiteState
from ._state import FuelRodFields, FuelRodState
from ._zones import zone_average


if TYPE_CHECKING:
    from ._problem import FuelRodAdmission


# Residual-row conditioning: each divisor brings one row block to order one.
THERMAL_ROW_SCALE_W: Final[float] = 1.0e4  # assembled heat-rate rows [W]
# Assembled force rows [N] are divided by this stress times a reference area
# [m^2]; displacement constraint rows [m] are multiplied by it so that they
# condition like force rows.
MECHANICAL_ROW_SCALE_PA: Final[float] = 1.0e11
WALL_TEMPERATURE_ROW_SCALE_K: Final[float] = 1.0e3  # prescribed wall rows [K]
WALL_HEAT_FLUX_ROW_SCALE_W_PER_M2: Final[float] = 1.0e6  # wall heat-flux rows
CHANNEL_ENERGY_ROW_SCALE_W: Final[float] = 1.0e6  # coupled channel energy rows
# Friction rows [N]; also the force scale of the Coulomb stick/slip map.
FRICTION_FORCE_ROW_SCALE_N: Final[float] = 1.0e3

# Physical admissibility allows these roundoff undershoots below zero.
GAP_VOLUME_ROUNDOFF_M3: Final[float] = 1.0e-12
RETAINED_GAS_ROUNDOFF_MOL: Final[float] = 1.0e-14

HYDROGEN_PPM_PER_MASS_FRACTION: Final[float] = 1.0e6

_GAS_SPECIES_COUNT: Final[int] = len(FUEL_GAS_SPECIES)
_HELIUM: Final[int] = FUEL_GAS_SPECIES.index("He")
_KRYPTON: Final[int] = FUEL_GAS_SPECIES.index("Kr")
_XENON: Final[int] = FUEL_GAS_SPECIES.index("Xe")


class FuelFormBinding(Protocol):
    """Prepared rod data read by the forms."""

    admission: FuelRodAdmission
    geometry: FuelRodGeometry
    heavy_metal: HeavyMetalReference
    free_volume: FreeVolumeModel | NumericalFreeVolumeModel
    prepared_id: str


def thermal_volume_residual(
    body: FuelBodyGeometry,
    temperature: Array,
    enthalpy_increment: Array,
    reference_density: Array,
    reference_conductivity: Array,
    reference_source: Array,
    dt: Array,
    /,
) -> Array:
    """Backward-Euler heat rows [W]: enthalpy rate minus source, plus conduction.

    ``enthalpy_increment`` [J/kg] is the endpoint-minus-accepted difference.
    """
    gradient = body.gradients(temperature)
    flux = (reference_conductivity @ gradient[..., None])[..., 0]
    return body.assemble(
        reference_density * enthalpy_increment / dt - reference_source, flux
    )


def mechanical_volume_residual(
    body: FuelBodyGeometry,
    first_piola: Array,
    /,
    *,
    reference_body_force: Array | None = None,
) -> Array:
    """Axisymmetric (r, z) momentum rows [N] of the first Piola stress ``P``.

    The hoop component enters the radial row as the value term ``P[1, 1] / r``.
    """
    force = (
        jnp.zeros((body.site_count, 2), dtype=first_piola.dtype)
        if reference_body_force is None
        else reference_body_force
    )
    value_dual = -force.at[:, 0].add(-first_piola[:, 1, 1] / body.site_coordinates[:, 0])
    gradient_dual = jnp.stack(
        (
            jnp.stack((first_piola[:, 0, 0], first_piola[:, 0, 2]), axis=-1),
            jnp.stack((first_piola[:, 2, 0], first_piola[:, 2, 2]), axis=-1),
        ),
        axis=1,
    )
    return body.assemble(value_dual, gradient_dual)


def deformation(
    body: FuelBodyGeometry, displacement: Array, axial_strain: Array, /
) -> Array:
    """Axisymmetric deformation gradient at every site of ``body``.

    A layered body carries radial displacement only; its axial stretch is the
    uniform axial strain of the site's layer.
    """
    if body.layered:
        displacement = jnp.concatenate(
            (displacement, jnp.zeros_like(displacement)), axis=-1
        )
    gradient = body.gradients(displacement)
    if body.layered:
        gradient = gradient.at[:, 1, 1].set(axial_strain[body.site_zone])
    values = body.values(displacement)
    return axisymmetric_deformation_gradient(
        gradient, values[:, 0], body.site_coordinates[:, 0]
    )


def pulled_conductivity(deformation_gradient: Array, conductivity: Array, /) -> Array:
    """Reference-configuration conductivity ``J k F^-1 F^-T`` on the (r, z) plane."""
    kinematics = finite_strain_kinematics(deformation_gradient)
    inverse = kinematics.inverse_deformation_gradient
    reference = (
        kinematics.jacobian[:, None, None]
        * conductivity[:, None, None]
        * (inverse @ jnp.swapaxes(inverse, -1, -2))
    )
    return reference[:, (0, 2), :][:, :, (0, 2)]


class FuelFormEvaluation(StrictModule):
    residual_fields: FuelRodFields
    candidate: FuelRodState
    local_converged: Array
    in_range: Array
    physical: Array
    contact_gap: Array
    friction_residual: Array
    gas_error: Array
    energy_error: Array
    accommodation_extrapolated: Array
    accommodation_source_comparison: Array
    source_support: Array


def _direct_coolant_fraction(bundle: FuelModelBundle, /) -> Array:
    """Fraction of fission power deposited directly in the coolant."""
    return 1 - bundle.deposition.fuel_fraction - bundle.deposition.cladding_fraction


def _plenum_temperature(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    conditions: IrradiationConditions,
    time: Array,
    dt: Array,
    linear_power: Array,
    coolant_heat_flux: Array,
    bundle: FuelModelBundle,
    /,
) -> Array:
    """Plenum gas temperature [K]: coolant outlet temperature plus the bundle offset.

    A prescribed wall uses its top-zone wall temperature as the outlet
    temperature; a single channel integrates wall and direct power up to the
    outlet enthalpy.
    """
    coolant, geometry = prepared.admission.coolant, prepared.geometry
    if isinstance(coolant, PrescribedWallTemperature):
        return coolant.wall_temperature(time + dt)[-1] + bundle.plenum_temperature.offset
    if isinstance(coolant, CoupledChannelCoolant):
        return (
            coolant.bulk_temperature(fields.channel_state, time + dt)[-1]
            + bundle.plenum_temperature.offset
        )
    if not isinstance(coolant, SingleChannelCoolant):
        raise TypeError("Unsupported plenum coolant-temperature binding.")
    cladding_outer_surface = geometry.cladding_outer
    displacement = cladding_outer_surface.values(fields.displacement_cladding)
    radius = cladding_outer_surface.reference_radius + displacement[:, 0]
    if geometry.layered:
        stretch = 1 + fields.axial_strain_cladding[cladding_outer_surface.zone]
    else:
        tangent = (
            cladding_outer_surface.axial_gradient(fields.displacement_cladding)
            .at[:, 1]
            .add(1.0)
        )
        stretch = jnp.linalg.norm(tangent, axis=-1)
    wall_power = jnp.sum(
        cladding_outer_surface.weight
        * radius
        / cladding_outer_surface.reference_radius
        * stretch
        * coolant_heat_flux
    )
    direct_power = _direct_coolant_fraction(bundle) * jnp.sum(
        linear_power * geometry.layer_lengths
    )
    outlet_enthalpy = coolant.bulk_enthalpy(
        inlet_temperature=conditions.coolant_inlet_temperature,
        pressure=conditions.coolant_pressure,
        mass_flux=conditions.coolant_mass_flux,
        cumulative_power_to_coolant=wall_power + direct_power,
    )
    return (
        coolant.bulk_temperature(
            pressure=conditions.coolant_pressure, specific_enthalpy=outlet_enthalpy
        )
        + bundle.plenum_temperature.offset
    )


class CladdingSurfaceResponse(StrictModule):
    state: CladdingSiteState
    load_fraction: Array
    mean_temperature: Array
    oxide_thickness: Array
    metal_reference_areal_mass: Array
    water_wall_temperature: Array


def _oxidation_surface(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    previous: FuelRodState,
    candidate: CladdingSiteState,
    cladding_deformation: Array,
    outer_measure_ratio: Array,
    coolant_heat_flux: Array,
    fast_flux: Array,
    dt: Array,
    bundle: FuelModelBundle,
    /,
) -> CladdingSurfaceResponse:
    """Waterside oxide growth, hydrogen pickup, and the load-bearing metal.

    Corrosion and pickup are per axial zone at the zone-mean outer temperature.
    The water-side wall temperature is the metal outer temperature less the
    drop across the oxide.
    """
    geometry, specification = prepared.geometry, prepared.admission.specification
    cladding_mesh, cladding_outer_surface = geometry.cladding, geometry.cladding_outer
    zone_count = geometry.layer_lengths.shape[0]
    outer_temperature = cladding_outer_surface.values(fields.temperature_cladding)
    current_area = jax.ops.segment_sum(
        cladding_outer_surface.weight * outer_measure_ratio,
        cladding_outer_surface.zone,
        zone_count,
    )
    reference_area = (
        2 * jnp.pi * specification.cladding.outer_radius * geometry.layer_lengths
    )
    mean_outer_temperature = zone_average(
        outer_temperature,
        cladding_outer_surface.weight * outer_measure_ratio,
        cladding_outer_surface.zone,
        zone_count,
    )
    previous_oxide = zone_average(
        previous.cladding_sites.oxide_thickness,
        cladding_mesh.site_weight,
        cladding_mesh.site_zone,
        zone_count,
    )
    oxide = bundle.corrosion.advance(
        previous_oxide, mean_outer_temperature, fast_flux, dt
    )
    oxygen_increment = bundle.corrosion.oxygen_areal_mass(oxide - previous_oxide)
    previous_hydrogen_mass = zone_average(
        previous.cladding_sites.hydrogen_areal_mass,
        cladding_mesh.site_weight,
        cladding_mesh.site_zone,
        zone_count,
    )
    hydrogen_mass = (
        previous_hydrogen_mass
        + bundle.hydrogen.absorbed_hydrogen_areal_mass(oxygen_increment)
        * current_area
        / reference_area
    )
    metal_loss = bundle.corrosion.metal_thickness_consumed(oxide)
    inverse = finite_strain_kinematics(cladding_deformation).inverse_deformation_gradient
    # Current-configuration metal loss pulled back along the radial direction.
    reference_loss = metal_loss[cladding_mesh.site_zone] * jnp.linalg.norm(
        inverse[:, 0, :], axis=-1
    )
    metal_reference_outer_radius = specification.cladding.outer_radius - reference_loss
    load_fraction = jnp.clip(
        (
            jnp.minimum(cladding_mesh.site_outer_radius, metal_reference_outer_radius)
            ** 2
            - cladding_mesh.site_inner_radius**2
        )
        / (cladding_mesh.site_outer_radius**2 - cladding_mesh.site_inner_radius**2),
        0.0,
        1.0,
    )
    reference_mass = jax.ops.segment_sum(
        ZIRCALOY_DENSITY * cladding_mesh.site_weight * load_fraction,
        cladding_mesh.site_zone,
        zone_count,
    )
    metal_areal_mass = reference_mass / reference_area
    hydrogen_ppm = HYDROGEN_PPM_PER_MASS_FRACTION * hydrogen_mass / metal_areal_mass
    site_zone = cladding_mesh.site_zone
    cladding_state = eqx.tree_at(
        lambda state: (
            state.oxide_thickness,
            state.hydrogen_ppm,
            state.hydrogen_areal_mass,
        ),
        candidate,
        (oxide[site_zone], hydrogen_ppm[site_zone], hydrogen_mass[site_zone]),
    )
    oxide_conductivity = bundle.oxide_conductivity.thermal_conductivity(
        mean_outer_temperature
    )
    water_wall_temperature = (
        outer_temperature
        - coolant_heat_flux
        * oxide[cladding_outer_surface.zone]
        / oxide_conductivity[cladding_outer_surface.zone]
    )
    return CladdingSurfaceResponse(
        state=cladding_state,
        load_fraction=load_fraction,
        mean_temperature=mean_outer_temperature,
        oxide_thickness=oxide,
        metal_reference_areal_mass=metal_areal_mass,
        water_wall_temperature=water_wall_temperature,
    )


class _MaterialResponses(NamedTuple):
    fuel_deformation: Array
    cladding_deformation: Array
    fuel_temperature: Array  # site values [K]
    cladding_temperature: Array
    previous_fuel_temperature: Array
    previous_cladding_temperature: Array
    fuel: FuelConstitutiveResponse
    cladding: CladdingConstitutiveResponse
    fuel_conductivity: Array  # site values [W/(m K)]
    cladding_conductivity: Array


class _Plenum(NamedTuple):
    response: PlenumResponse
    mechanical_relocation: Array  # zone-mean relocation strain taken up by mechanics
    thermal_relocation: Array  # zone-mean relocation strain seen by the gap


class _InterfaceKinematics(NamedTuple):
    """Current fuel/cladding interface and cladding outer-surface geometry."""

    fuel_displacement: Array  # displacement at fuel interface points [m]
    cladding_displacement: Array
    fuel_radius: Array  # current radius [m]
    cladding_radius: Array
    fuel_tangent: Array  # current axial tangent, unit reference length
    cladding_tangent: Array
    contact: AxisymmetricInterfaceGeometry
    outer_measure_ratio: Array  # current/reference outer-surface measure
    outer_normal: Array


class _GapHeatTransfer(NamedTuple):
    law: GapConductance
    fuel_surface_temperature: Array
    cladding_surface_temperature: Array
    thermal_radius: Array  # fuel radius seen by gap heat transfer [m]
    heat_flux: Array  # fuel-to-cladding gap heat flux [W/m^2]


class _StoredEnthalpy(NamedTuple):
    fuel_density: Array  # reference density [kg/m^3]
    cladding_density: Array
    fuel_increment: Array  # endpoint-minus-accepted specific enthalpy [J/kg]
    cladding_increment: Array


class _CoolantRows(NamedTuple):
    flux: Array  # rows of the coolant_heat_flux unknowns
    channel: Array  # rows of the channel_state unknowns
    admissible: Array
    source_supported: Array


class _MechanicalRows(NamedTuple):
    fuel: Array
    cladding: Array
    fuel_axial: Array
    cladding_axial: Array
    slip: Array  # fuel-minus-cladding axial slip this step, per contact zone [m]


def _unit_normal(tangent: Array, /) -> Array:
    """Outward radial normal of an (r, z) curve with axial tangent ``tangent``."""
    return (
        jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
        / jnp.linalg.norm(tangent, axis=-1)[:, None]
    )


def _zone_midpoint_axial_displacement(
    layer_lengths: Array, axial_strain: Array, /
) -> Array:
    """Axial displacement [m] at each layer midpoint of a layered body."""
    return jnp.cumsum(layer_lengths * axial_strain) - 0.5 * layer_lengths * axial_strain


def _material_responses(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    previous: FuelRodState,
    dt: Array,
    bundle: FuelModelBundle,
    frozen_gas: FuelGasBlockResult,
    loading: FuelStepLoading,
    /,
) -> _MaterialResponses:
    """Local constitutive returns and conductivities at every fuel and cladding site."""
    geometry, specification = prepared.geometry, prepared.admission.specification
    fuel_mesh, cladding_mesh = geometry.fuel, geometry.cladding
    fuel_deformation = deformation(
        fuel_mesh, fields.displacement_fuel, fields.axial_strain_fuel
    )
    cladding_deformation = deformation(
        cladding_mesh, fields.displacement_cladding, fields.axial_strain_cladding
    )
    previous_cladding_deformation = deformation(
        cladding_mesh,
        previous.fields.displacement_cladding,
        previous.fields.axial_strain_cladding,
    )
    fuel_temperature = fuel_mesh.values(fields.temperature_fuel)
    cladding_temperature = cladding_mesh.values(fields.temperature_cladding)
    previous_fuel_temperature = fuel_mesh.values(previous.fields.temperature_fuel)
    previous_cladding_temperature = cladding_mesh.values(
        previous.fields.temperature_cladding
    )

    def fuel_site(
        accepted: FuelSiteState,
        site_deformation: Array,
        temperature: Array,
        burnup: Array,
        rod_burnup: Array,
        linear_power: Array,
        fission_rate: Array,
        site_gas: FuelGasBlockResult,
    ) -> FuelConstitutiveResponse:
        return fuel_response(
            bundle,
            specification,
            accepted,
            site_deformation,
            temperature,
            burnup,
            rod_burnup,
            linear_power,
            fission_rate,
            dt,
            site_gas,
        )

    def cladding_site(
        accepted: CladdingSiteState,
        site_deformation: Array,
        previous_site_deformation: Array,
        temperature: Array,
        fast_flux: Array,
    ) -> CladdingConstitutiveResponse:
        return cladding_response(
            bundle,
            specification,
            accepted,
            site_deformation,
            previous_site_deformation,
            temperature,
            fast_flux,
            dt,
        )

    fuel = jax.vmap(fuel_site)(
        previous.fuel_sites,
        fuel_deformation,
        fuel_temperature,
        loading.exposure.burnup_mwd_per_kghm,
        loading.radial_power.burnup_mwd_per_kghm[fuel_mesh.site_zone],
        loading.conditions.linear_power[fuel_mesh.site_zone],
        loading.fission_rate,
        frozen_gas,
    )
    cladding = jax.vmap(cladding_site)(
        previous.cladding_sites,
        cladding_deformation,
        previous_cladding_deformation,
        cladding_temperature,
        loading.fast_flux[cladding_mesh.site_zone],
    )
    fuel_conductivity = bundle.fuel_conductivity.conductivity(
        fuel_temperature,
        loading.exposure.burnup_mwd_per_kghm,
        fuel.conductivity_porosity,
        0.0,  # gadolinia weight fraction of undoped UO2
    )
    cladding_conductivity = bundle.cladding_conductivity.thermal_conductivity(
        cladding_temperature
    )
    return _MaterialResponses(
        fuel_deformation=fuel_deformation,
        cladding_deformation=cladding_deformation,
        fuel_temperature=fuel_temperature,
        cladding_temperature=cladding_temperature,
        previous_fuel_temperature=previous_fuel_temperature,
        previous_cladding_temperature=previous_cladding_temperature,
        fuel=fuel,
        cladding=cladding,
        fuel_conductivity=fuel_conductivity,
        cladding_conductivity=cladding_conductivity,
    )


def _plenum(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    time: Array,
    dt: Array,
    bundle: FuelModelBundle,
    loading: FuelStepLoading,
    material: _MaterialResponses,
    coolant_heat_flux: Array,
    /,
) -> _Plenum:
    """Free-gas volumes and spring force from zone means of the fuel response."""
    geometry = prepared.geometry
    fuel_mesh = geometry.fuel
    zone_count = geometry.layer_lengths.shape[0]
    fuel = material.fuel

    def fuel_zone_mean(values: Array, /) -> Array:
        return zone_average(
            values, fuel_mesh.site_weight, fuel_mesh.site_zone, zone_count
        )

    mechanical_relocation = fuel_zone_mean(fuel.relocation_strain)
    thermal_relocation = fuel_zone_mean(fuel.thermal_relocation_strain)
    densification_loss = fuel_zone_mean(
        1 - (1 + fuel.state.mechanics.densification_strain) ** 3
    )
    mean_fuel_temperature = fuel_zone_mean(material.fuel_temperature)
    connected_porosity = fuel_zone_mean(fuel.state.gas.total.connected_boundary_volume)
    thermal_stretch = fuel_zone_mean(
        1 + bundle.fuel_expansion.linear_strain(material.fuel_temperature)
    )
    plenum_temperature = _plenum_temperature(
        prepared,
        fields,
        loading.conditions,
        time,
        dt,
        loading.linear_power,
        coolant_heat_flux,
        bundle,
    )
    response = plenum_response(
        prepared.admission.specification,
        geometry,
        prepared.free_volume,
        fields,
        mean_fuel_temperature,
        thermal_relocation,
        densification_loss,
        connected_porosity,
        plenum_temperature,
        mechanical_relocation,
        thermal_stretch,
    )
    return _Plenum(
        response=response,
        mechanical_relocation=mechanical_relocation,
        thermal_relocation=thermal_relocation,
    )


def _interface_kinematics(
    geometry: FuelRodGeometry, fields: FuelRodFields, plenum: PlenumResponse, /
) -> _InterfaceKinematics:
    """Current interface traces, contact geometry, and outer-surface measure.

    A layered rod's traces stay at their reference height and stretch with the
    zone-mean axial strain; RZ traces follow the full displacement field.
    """
    fuel_interface = geometry.fuel_interface
    cladding_interface = geometry.cladding_interface
    cladding_outer_surface = geometry.cladding_outer
    fuel_displacement = fuel_interface.values(fields.displacement_fuel)
    cladding_displacement = cladding_interface.values(fields.displacement_cladding)
    outer_displacement = cladding_outer_surface.values(fields.displacement_cladding)
    fuel_radius = fuel_interface.reference_radius + fuel_displacement[:, 0]
    cladding_radius = cladding_interface.reference_radius + cladding_displacement[:, 0]
    outer_radius = cladding_outer_surface.reference_radius + outer_displacement[:, 0]
    if geometry.layered:
        fuel_tangent = jnp.stack(
            (
                jnp.zeros_like(fuel_radius),
                1 + plenum.axial_strain_fuel[fuel_interface.zone],
            ),
            axis=-1,
        )
        cladding_tangent = jnp.stack(
            (
                jnp.zeros_like(cladding_radius),
                1 + plenum.axial_strain_cladding[cladding_interface.zone],
            ),
            axis=-1,
        )
        fuel_position = jnp.stack((fuel_radius, fuel_interface.reference_z), axis=-1)
        cladding_position = jnp.stack(
            (cladding_radius, cladding_interface.reference_z), axis=-1
        )
        outer_measure_ratio = (
            outer_radius
            / cladding_outer_surface.reference_radius
            * (1 + plenum.axial_strain_cladding[cladding_outer_surface.zone])
        )
        outer_normal = jnp.broadcast_to(jnp.asarray((1.0, 0.0)), fuel_tangent.shape)
    else:
        fuel_tangent = (
            fuel_interface.axial_gradient(fields.displacement_fuel).at[:, 1].add(1.0)
        )
        cladding_tangent = (
            cladding_interface.axial_gradient(fields.displacement_cladding)
            .at[:, 1]
            .add(1.0)
        )
        fuel_position = jnp.stack(
            (fuel_radius, fuel_interface.reference_z + fuel_displacement[:, 1]), axis=-1
        )
        cladding_position = jnp.stack(
            (
                cladding_radius,
                cladding_interface.reference_z + cladding_displacement[:, 1],
            ),
            axis=-1,
        )
        outer_tangent = (
            cladding_outer_surface.axial_gradient(fields.displacement_cladding)
            .at[:, 1]
            .add(1.0)
        )
        outer_measure_ratio = (
            outer_radius
            / cladding_outer_surface.reference_radius
            * jnp.linalg.norm(outer_tangent, axis=-1)
        )
        outer_normal = _unit_normal(outer_tangent)
    contact = axisymmetric_interface_geometry(
        fuel_position,
        cladding_position,
        fuel_tangent,
        cladding_tangent,
        fuel_interface.reference_radius,
    )
    return _InterfaceKinematics(
        fuel_displacement=fuel_displacement,
        cladding_displacement=cladding_displacement,
        fuel_radius=fuel_radius,
        cladding_radius=cladding_radius,
        fuel_tangent=fuel_tangent,
        cladding_tangent=cladding_tangent,
        contact=contact,
        outer_measure_ratio=outer_measure_ratio,
        outer_normal=outer_normal,
    )


def _gas_ledger(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    previous: FuelRodState,
    dt: Array,
    loading: FuelStepLoading,
    fuel: FuelConstitutiveResponse,
    /,
) -> FuelGasLedger:
    """Rod gas inventory [mol] with the candidate's free gas as released gas."""
    fuel_mesh = prepared.geometry.fuel
    material_fraction = (
        1.0 - prepared.admission.specification.pellet.removed_volume_fraction
    )
    produced_moles = (
        previous.gas.produced_moles
        + fuel_mesh.integrate(dt * loading.produced_species)
        * material_fraction
        / AVOGADRO_PER_MOLE
    )
    retained_moles = (
        fuel_mesh.integrate(fuel.state.gas.retained_species)
        * material_fraction
        / AVOGADRO_PER_MOLE
    )
    released_moles = fields.free_moles - previous.gas.fill_moles
    return FuelGasLedger(
        fill_moles=previous.gas.fill_moles,
        produced_moles=produced_moles,
        retained_moles=retained_moles,
        released_moles=released_moles,
        exported_moles=previous.gas.exported_moles,
        balance_error=produced_moles
        - retained_moles
        - released_moles
        - previous.gas.exported_moles,
    )


def _gap_heat_transfer(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    bundle: FuelModelBundle,
    material: _MaterialResponses,
    plenum: _Plenum,
    interface: _InterfaceKinematics,
    /,
) -> _GapHeatTransfer:
    """Gap conductance across the relocation-corrected thermal gap."""
    geometry, specification = prepared.geometry, prepared.admission.specification
    fuel_mesh, cladding_mesh = geometry.fuel, geometry.cladding
    fuel_interface = geometry.fuel_interface
    zone_count = geometry.layer_lengths.shape[0]
    fuel_surface_temperature = fuel_interface.values(fields.temperature_fuel)
    cladding_surface_temperature = geometry.cladding_interface.values(
        fields.temperature_cladding
    )
    composition = fields.free_moles / jnp.sum(fields.free_moles)
    fuel_zone_conductivity = zone_average(
        material.fuel_conductivity, fuel_mesh.site_weight, fuel_mesh.site_zone, zone_count
    )
    cladding_zone_conductivity = zone_average(
        material.cladding_conductivity,
        cladding_mesh.site_weight,
        cladding_mesh.site_zone,
        zone_count,
    )
    # Relocation counted by the gap model but not yet by the mechanical
    # relocation strain narrows the thermal gap only [m].
    thermal_only_relocation = (
        specification.pellet.radius
        * (plenum.thermal_relocation - plenum.mechanical_relocation)[fuel_interface.zone]
    )
    thermal_gap = jnp.maximum(interface.contact.normal_gap - thermal_only_relocation, 0.0)
    thermal_radius = jnp.minimum(
        interface.fuel_radius + thermal_only_relocation, interface.cladding_radius
    )
    law = bundle.gap.conductance(
        fuel_surface_temperature=fuel_surface_temperature,
        cladding_surface_temperature=cladding_surface_temperature,
        gap_width=thermal_gap,
        contact_pressure=fields.contact_pressure[fuel_interface.interface_index],
        gas_mole_fractions=composition,
        gas_pressure=fields.pressure,
        fuel_roughness=specification.pellet.roughness,
        cladding_roughness=specification.cladding.roughness,
        fuel_emissivity=bundle.fuel_emissivity.emissivity(fuel_surface_temperature),
        cladding_emissivity=bundle.bore_emissivity.value,
        fuel_radius=thermal_radius,
        cladding_inner_radius=interface.cladding_radius,
        fuel_conductivity=fuel_zone_conductivity[fuel_interface.zone],
        cladding_conductivity=cladding_zone_conductivity[fuel_interface.zone],
    )
    return _GapHeatTransfer(
        law=law,
        fuel_surface_temperature=fuel_surface_temperature,
        cladding_surface_temperature=cladding_surface_temperature,
        thermal_radius=thermal_radius,
        heat_flux=law.total * (fuel_surface_temperature - cladding_surface_temperature),
    )


def _stored_enthalpy(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    bundle: FuelModelBundle,
    material: _MaterialResponses,
    /,
) -> _StoredEnthalpy:
    """Reference densities and the step's specific-enthalpy increments."""
    specification = prepared.admission.specification
    material_fraction = 1.0 - specification.pellet.removed_volume_fraction
    # Fuel density is smeared over the pellet volume removed by dishes/chamfers.
    fuel_density = jnp.asarray(
        specification.pellet.density_fraction
        * UO2_THEORETICAL_DENSITY
        * material_fraction,
        dtype=fields.pressure.dtype,
    )
    cladding_density = jnp.asarray(ZIRCALOY_DENSITY, dtype=fields.pressure.dtype)
    fuel_increment = bundle.fuel_heat.enthalpy(
        material.fuel_temperature
    ) - bundle.fuel_heat.enthalpy(material.previous_fuel_temperature)
    cladding_increment = bundle.cladding_heat.specific_enthalpy(
        material.cladding_temperature
    ) - bundle.cladding_heat.specific_enthalpy(material.previous_cladding_temperature)
    return _StoredEnthalpy(
        fuel_density=fuel_density,
        cladding_density=cladding_density,
        fuel_increment=fuel_increment,
        cladding_increment=cladding_increment,
    )


def _thermal_rows(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    previous: FuelRodState,
    dt: Array,
    bundle: FuelModelBundle,
    loading: FuelStepLoading,
    material: _MaterialResponses,
    enthalpy: _StoredEnthalpy,
    gap: _GapHeatTransfer,
    interface: _InterfaceKinematics,
    coolant_heat_flux: Array,
    /,
) -> tuple[Array, Array]:
    """Fuel and cladding heat rows [W] with gap and coolant heat exchange."""
    geometry, specification = prepared.geometry, prepared.admission.specification
    fuel_mesh, cladding_mesh = geometry.fuel, geometry.cladding
    fuel_interface = geometry.fuel_interface
    cladding_interface = geometry.cladding_interface
    linear_power = loading.linear_power
    fuel_area = jnp.pi * specification.pellet.radius**2
    cladding_area = jnp.pi * (
        specification.cladding.outer_radius**2 - specification.cladding.inner_radius**2
    )
    fuel_source = (
        bundle.deposition.fuel_fraction
        * linear_power[fuel_mesh.site_zone]
        * loading.radial_shape
        / fuel_area
    )
    cladding_source = (
        bundle.deposition.cladding_fraction
        * linear_power[cladding_mesh.site_zone]
        / cladding_area
    )
    fuel_rows = thermal_volume_residual(
        fuel_mesh,
        fields.temperature_fuel,
        enthalpy.fuel_increment,
        enthalpy.fuel_density,
        pulled_conductivity(material.fuel_deformation, material.fuel_conductivity),
        fuel_source,
        dt,
    )
    cladding_rows = thermal_volume_residual(
        cladding_mesh,
        fields.temperature_cladding,
        enthalpy.cladding_increment,
        enthalpy.cladding_density,
        pulled_conductivity(
            material.cladding_deformation, material.cladding_conductivity
        ),
        cladding_source,
        dt,
    )
    # Gap heat crosses the interface at the relocation-corrected fuel radius.
    heat_measure = (
        interface.contact.current_measure_ratio
        * gap.thermal_radius
        / interface.fuel_radius
    )
    fuel_rows = fuel_rows + fuel_interface.assemble(heat_measure * gap.heat_flux)
    cladding_rows = cladding_rows - cladding_interface.assemble(
        heat_measure * fuel_interface.weight / cladding_interface.weight * gap.heat_flux
    )
    cladding_rows = cladding_rows + geometry.cladding_outer.assemble(
        interface.outer_measure_ratio * coolant_heat_flux
    )
    return fuel_rows, cladding_rows


def _coolant_rows(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    previous: FuelRodState,
    conditions: IrradiationConditions,
    time: Array,
    dt: Array,
    outer_measure_ratio: Array,
    coolant_heat_flux: Array,
    water_wall_temperature: Array,
    wall_power: Array,
    direct_coolant_power: Array,
    /,
) -> _CoolantRows:
    """Water-side heat-flux closure rows for the admitted coolant model."""
    geometry = prepared.geometry
    cladding_outer_surface = geometry.cladding_outer
    zone_count = geometry.layer_lengths.shape[0]
    channel_rows = jnp.zeros_like(fields.channel_state)
    coolant = prepared.admission.coolant
    if isinstance(coolant, PrescribedWallTemperature):
        flux_rows = (
            water_wall_temperature
            - coolant.wall_temperature(time + dt)[cladding_outer_surface.zone]
        ) / WALL_TEMPERATURE_ROW_SCALE_K
        admissible = jnp.asarray(True)
        source_supported = jnp.asarray(True)
    elif isinstance(coolant, SingleChannelCoolant):
        zone_power = wall_power + direct_coolant_power
        # Power delivered to the coolant upstream of each zone midpoint [W].
        cumulative_power = jnp.cumsum(zone_power) - 0.5 * zone_power
        enthalpy = coolant.bulk_enthalpy(
            inlet_temperature=conditions.coolant_inlet_temperature,
            pressure=conditions.coolant_pressure,
            mass_flux=conditions.coolant_mass_flux,
            cumulative_power_to_coolant=cumulative_power,
        )
        bulk_temperature = coolant.bulk_temperature(
            pressure=conditions.coolant_pressure, specific_enthalpy=enthalpy
        )
        predicted_flux = coolant.wall_heat_flux(
            wall_temperature=water_wall_temperature,
            bulk_temperature=bulk_temperature[cladding_outer_surface.zone],
            pressure=conditions.coolant_pressure,
            mass_flux=conditions.coolant_mass_flux,
        )
        flux_rows = (
            coolant_heat_flux - predicted_flux
        ) / WALL_HEAT_FLUX_ROW_SCALE_W_PER_M2
        admissible = jnp.all(
            coolant.admissible(
                pressure=conditions.coolant_pressure,
                mass_flux=conditions.coolant_mass_flux,
                specific_enthalpy=enthalpy,
            )
        )
        source_supported = jnp.all(
            coolant.nucleate_boiling_supported(
                wall_temperature=water_wall_temperature,
                bulk_temperature=bulk_temperature[cladding_outer_surface.zone],
                pressure=conditions.coolant_pressure,
                heat_flux=predicted_flux,
            )
        )
    elif isinstance(coolant, CoupledChannelCoolant):
        areas = jax.ops.segment_sum(
            cladding_outer_surface.weight * outer_measure_ratio,
            cladding_outer_surface.zone,
            zone_count,
        )
        average_wall_temperature = zone_average(
            water_wall_temperature,
            cladding_outer_surface.weight * outer_measure_ratio,
            cladding_outer_surface.zone,
            zone_count,
        )
        average_flux = wall_power / areas
        channel_rows = (
            coolant.residual(
                fields.channel_state,
                previous.fields.channel_state,
                average_wall_temperature,
                average_flux,
                time + dt,
                dt,
                deposited_direct_power=direct_coolant_power,
                interface_areas=areas,
            )
            / CHANNEL_ENERGY_ROW_SCALE_W
        )
        predicted_flux = coolant.wall_heat_flux(
            fields.channel_state, average_wall_temperature, time + dt
        )
        flux_rows = (
            coolant_heat_flux - predicted_flux[cladding_outer_surface.zone]
        ) / WALL_HEAT_FLUX_ROW_SCALE_W_PER_M2
        admissible = coolant.admissible(fields.channel_state, time + dt)
        source_supported = jnp.all(
            coolant.wall_transfer.nucleate_boiling_supported(
                wall_temperature=average_wall_temperature,
                bulk_temperature=coolant.bulk_temperature(
                    fields.channel_state, time + dt
                ),
                pressure=conditions.coolant_pressure,
                heat_flux=predicted_flux,
            )
        )
    else:
        raise TypeError("Unsupported admitted coolant boundary.")
    return _CoolantRows(
        flux=geometry.coolant_flux_rows(flux_rows, outer_measure_ratio),
        channel=channel_rows,
        admissible=admissible,
        source_supported=source_supported,
    )


def _axial_rows(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    previous: FuelRodState,
    conditions: IrradiationConditions,
    plenum: PlenumResponse,
    material: _MaterialResponses,
    load_fraction: Array,
    interface: _InterfaceKinematics,
    fuel_rows: Array,
    cladding_rows: Array,
    /,
) -> _MechanicalRows:
    """Axial equilibrium and the axial slip that drives interface friction.

    A layered rod has one axial strain per layer: axial equilibrium is one row
    per layer and the radial momentum rows keep their radial component only.
    An RZ rod carries the plenum pressure and spring force as end tractions and
    pins one axial displacement per body.
    """
    geometry, specification = prepared.geometry, prepared.admission.specification
    fuel_mesh, cladding_mesh = geometry.fuel, geometry.cladding
    fuel_interface = geometry.fuel_interface
    zone_count = geometry.layer_lengths.shape[0]
    fuel_area = jnp.pi * specification.pellet.radius**2
    cladding_area = jnp.pi * (
        specification.cladding.outer_radius**2 - specification.cladding.inner_radius**2
    )
    if geometry.layered:
        fuel_axial_force = (
            jax.ops.segment_sum(
                material.fuel.stress.first_piola[:, 2, 2] * fuel_mesh.site_weight,
                fuel_mesh.site_zone,
                zone_count,
            )
            / geometry.layer_lengths
        )
        cladding_axial_force = (
            jax.ops.segment_sum(
                material.cladding.stress.first_piola[:, 2, 2]
                * load_fraction
                * cladding_mesh.site_weight,
                cladding_mesh.site_zone,
                zone_count,
            )
            / geometry.layer_lengths
        )
        # Friction carried by this layer and every layer above it [N].
        friction_from_above = jnp.cumsum(fields.friction_force[::-1])[::-1]
        fuel_axial_rows = (
            fuel_axial_force
            + fields.pressure * jnp.pi * plenum.fuel_surface_radius**2
            + plenum.spring_force
            - friction_from_above
        ) / (MECHANICAL_ROW_SCALE_PA * fuel_area)
        cladding_axial_rows = (
            cladding_axial_force
            - fields.pressure * jnp.pi * plenum.cladding_inner_radius**2
            + conditions.coolant_pressure * jnp.pi * plenum.cladding_outer_radius**2
            - plenum.spring_force
            + friction_from_above
        ) / (MECHANICAL_ROW_SCALE_PA * cladding_area)
        slip = (
            _zone_midpoint_axial_displacement(
                geometry.layer_lengths, fields.axial_strain_fuel
            )
            - _zone_midpoint_axial_displacement(
                geometry.layer_lengths, fields.axial_strain_cladding
            )
            - _zone_midpoint_axial_displacement(
                geometry.layer_lengths, previous.fields.axial_strain_fuel
            )
            + _zone_midpoint_axial_displacement(
                geometry.layer_lengths, previous.fields.axial_strain_cladding
            )
        )
        return _MechanicalRows(
            fuel=fuel_rows[:, :1],
            cladding=cladding_rows[:, :1],
            fuel_axial=fuel_axial_rows,
            cladding_axial=cladding_axial_rows,
            slip=slip,
        )
    if geometry.fuel_end is None or geometry.cladding_end is None:
        raise ValueError("RZ geometry requires fuel and cladding end traces.")
    fuel_end, cladding_end = geometry.fuel_end, geometry.cladding_end
    fuel_end_area = jnp.pi * plenum.fuel_surface_radius[-1] ** 2
    fuel_end_load = -fields.pressure * fuel_end_area - plenum.spring_force
    cladding_end_load = (
        fields.pressure * jnp.pi * plenum.cladding_inner_radius[-1] ** 2
        - conditions.coolant_pressure * jnp.pi * plenum.cladding_outer_radius[-1] ** 2
        + plenum.spring_force
    )
    # End loads [N] act as uniform axial tractions over the reference sections.
    fuel_rows = fuel_rows - fuel_end.assemble(
        jnp.broadcast_to(
            jnp.stack((jnp.asarray(0.0), fuel_end_load / fuel_area)),
            (fuel_end.weight.shape[0], 2),
        )
    )
    cladding_rows = cladding_rows - cladding_end.assemble(
        jnp.broadcast_to(
            jnp.stack((jnp.asarray(0.0), cladding_end_load / cladding_area)),
            (cladding_end.weight.shape[0], 2),
        )
    )
    fuel_rows = fuel_rows.at[geometry.axial_anchor_fuel, 1].set(
        fields.displacement_fuel[geometry.axial_anchor_fuel, 1] * MECHANICAL_ROW_SCALE_PA
    )
    cladding_rows = cladding_rows.at[geometry.axial_anchor_cladding, 1].set(
        fields.displacement_cladding[geometry.axial_anchor_cladding, 1]
        * MECHANICAL_ROW_SCALE_PA
    )
    previous_fuel_displacement = fuel_interface.values(previous.fields.displacement_fuel)
    previous_cladding_displacement = geometry.cladding_interface.values(
        previous.fields.displacement_cladding
    )
    point_slip = jnp.sum(
        (
            interface.fuel_displacement
            - interface.cladding_displacement
            - previous_fuel_displacement
            + previous_cladding_displacement
        )
        * interface.contact.tangent,
        axis=-1,
    )
    dtype = fields.temperature_fuel.dtype
    return _MechanicalRows(
        fuel=fuel_rows,
        cladding=cladding_rows,
        fuel_axial=jnp.zeros((0,), dtype=dtype),
        cladding_axial=jnp.zeros((0,), dtype=dtype),
        slip=zone_average(
            point_slip,
            fuel_interface.weight,
            fuel_interface.interface_index,
            fields.friction_force.shape[0],
        ),
    )


def _mechanical_rows(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    previous: FuelRodState,
    conditions: IrradiationConditions,
    plenum: PlenumResponse,
    material: _MaterialResponses,
    load_fraction: Array,
    interface: _InterfaceKinematics,
    /,
) -> _MechanicalRows:
    """Momentum rows [N]: stress, contact/friction traction, gas and coolant pressure.

    Only the load-bearing metal fraction of the corroded cladding carries stress.
    """
    geometry = prepared.geometry
    fuel_interface = geometry.fuel_interface
    cladding_interface = geometry.cladding_interface
    contact = interface.contact
    common_measure = contact.current_measure_ratio
    paired_measure = common_measure * fuel_interface.weight / cladding_interface.weight
    fuel_rows = mechanical_volume_residual(
        geometry.fuel, material.fuel.stress.first_piola
    )
    cladding_rows = mechanical_volume_residual(
        geometry.cladding,
        material.cladding.stress.first_piola * load_fraction[:, None, None],
    )
    contact_pressure = fields.contact_pressure[fuel_interface.interface_index]
    friction_force = fields.friction_force[fuel_interface.interface_index]
    interface_area = (
        2
        * jnp.pi
        * interface.fuel_radius
        * geometry.interface_lengths[fuel_interface.interface_index]
        * jnp.linalg.norm(interface.fuel_tangent, axis=-1)
    )
    fuel_traction = (
        -contact_pressure[:, None] * contact.normal
        + (friction_force / interface_area)[:, None] * contact.tangent
    )
    cladding_traction = -fuel_traction
    fuel_rows = fuel_rows - fuel_interface.assemble(
        common_measure[:, None] * fuel_traction
    )
    cladding_rows = cladding_rows - cladding_interface.assemble(
        paired_measure[:, None] * cladding_traction
    )
    cladding_inner_measure = (
        interface.cladding_radius
        / cladding_interface.reference_radius
        * jnp.linalg.norm(interface.cladding_tangent, axis=-1)
    )
    fuel_rows = fuel_rows + fuel_interface.assemble(
        common_measure[:, None] * fields.pressure * _unit_normal(interface.fuel_tangent)
    )
    cladding_rows = cladding_rows - cladding_interface.assemble(
        cladding_inner_measure[:, None]
        * fields.pressure
        * _unit_normal(interface.cladding_tangent)
    )
    cladding_rows = cladding_rows + geometry.cladding_outer.assemble(
        interface.outer_measure_ratio[:, None]
        * conditions.coolant_pressure
        * interface.outer_normal
    )
    rows = _axial_rows(
        prepared,
        fields,
        previous,
        conditions,
        plenum,
        material,
        load_fraction,
        interface,
        fuel_rows,
        cladding_rows,
    )
    # Radial displacement vanishes on the fuel axis.
    return rows._replace(
        fuel=rows.fuel.at[geometry.fuel_axis_nodes, 0].set(
            fields.displacement_fuel[geometry.fuel_axis_nodes, 0]
            * MECHANICAL_ROW_SCALE_PA
        )
    )


def _friction_rows(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    bundle: FuelModelBundle,
    interface: _InterfaceKinematics,
    slip: Array,
    /,
) -> Array:
    """Interface friction rows: zero force when frictionless, else Coulomb stick/slip."""
    match prepared.admission.axial_interaction:
        case "frictionless":
            return fields.friction_force / FRICTION_FORCE_ROW_SCALE_N
        case "coulomb":
            friction = bundle.friction
            if friction is None:
                raise ValueError("Coulomb interaction requires source-bound friction.")
            fuel_interface = prepared.geometry.fuel_interface
            normal_force = fields.contact_pressure * jax.ops.segment_sum(
                fuel_interface.weight * interface.contact.current_measure_ratio,
                fuel_interface.interface_index,
                fields.contact_pressure.shape[0],
            )
            return coulomb_interface_response(
                fields.friction_force,
                slip,
                normal_force,
                friction.coefficient,
                force_scale=FRICTION_FORCE_ROW_SCALE_N,
                slip_scale=prepared.admission.specification.rod.stack_length,
            ).residual
        case unsupported:
            assert_never(unsupported)


def _energy_ledger(
    prepared: FuelFormBinding,
    previous: FuelRodState,
    dt: Array,
    bundle: FuelModelBundle,
    loading: FuelStepLoading,
    enthalpy: _StoredEnthalpy,
    wall_power: Array,
    direct_coolant_power: Array,
    /,
) -> FuelEnergyLedger:
    """Cumulative stored, deposited and delivered energy [J] at the endpoint."""
    geometry = prepared.geometry
    rod_power = jnp.sum(loading.linear_power * geometry.layer_lengths)
    stored_fuel = previous.energy.stored_fuel + geometry.fuel.integrate(
        enthalpy.fuel_density * enthalpy.fuel_increment
    )
    stored_cladding = previous.energy.stored_cladding + geometry.cladding.integrate(
        enthalpy.cladding_density * enthalpy.cladding_increment
    )
    deposited_fuel = (
        previous.energy.deposited_fuel + bundle.deposition.fuel_fraction * rod_power * dt
    )
    deposited_cladding = (
        previous.energy.deposited_cladding
        + bundle.deposition.cladding_fraction * rod_power * dt
    )
    deposited_direct_coolant = (
        previous.energy.deposited_direct_coolant + jnp.sum(direct_coolant_power) * dt
    )
    delivered_coolant = (
        previous.energy.delivered_coolant
        + jnp.sum(wall_power) * dt
        + jnp.sum(direct_coolant_power) * dt
    )
    return FuelEnergyLedger(
        deposited_fuel=deposited_fuel,
        deposited_cladding=deposited_cladding,
        deposited_direct_coolant=deposited_direct_coolant,
        stored_fuel=stored_fuel,
        stored_cladding=stored_cladding,
        delivered_coolant=delivered_coolant,
        balance_error=stored_fuel
        + stored_cladding
        + delivered_coolant
        - deposited_fuel
        - deposited_cladding
        - deposited_direct_coolant,
    )


def _physically_admissible(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    material: _MaterialResponses,
    plenum: PlenumResponse,
    gas: FuelGasLedger,
    gap: _GapHeatTransfer,
    interface: _InterfaceKinematics,
    surface: CladdingSurfaceResponse,
    coolant_admissible: Array,
    /,
) -> Array:
    """Whether every constitutive, gas, volume, contact and oxide state is physical."""
    cladding = prepared.admission.specification.cladding
    fuel = material.fuel
    return (
        jnp.all(fuel.stress.admissible)
        & jnp.all(material.cladding.stress.admissible)
        & jnp.all(fuel.gas.admissible)
        & jnp.all(fuel.species.admissible)
        & jnp.all(fuel.helium.admissible)
        & jnp.all(gap.law.successful)
        & (fields.pressure > 0)
        & (plenum.components.plenum_volume > 0)
        & jnp.all(plenum.components.gap_volume >= -GAP_VOLUME_ROUNDOFF_M3)
        & jnp.all(gas.retained_moles >= -RETAINED_GAS_ROUNDOFF_MOL)
        & jnp.all(fields.free_moles >= 0)
        & coolant_admissible
        & jnp.all(interface.contact.admissible)
        & jnp.all(surface.metal_reference_areal_mass > 0)
        & jnp.all(surface.oxide_thickness < cladding.outer_radius - cladding.inner_radius)
        & jnp.all((fuel.conductivity_porosity >= 0) & (fuel.conductivity_porosity < 1))
    )


def _source_support(
    bundle: FuelModelBundle,
    material: _MaterialResponses,
    gap: _GapHeatTransfer,
    mean_outer_temperature: Array,
    coolant_source_supported: Array,
    /,
) -> Array:
    """Whether each source law is inside its validity range, in ``SOURCE_SLOTS`` order."""
    fuel_support = jnp.all(material.fuel.source_support, axis=0)
    cladding_support = jnp.all(material.cladding.source_support, axis=0)
    if fuel_support.shape != (len(FUEL_SITE_SOURCE_SLOTS),) or (
        cladding_support.shape != (len(CLADDING_SITE_SOURCE_SLOTS),)
    ):
        raise RuntimeError(
            "Constitutive source support must have one column per source slot."
        )
    bore_low, bore_high = bundle.bore_emissivity.temperature_range
    bore_temperature = gap.cladding_surface_temperature
    rod_support = {
        "gap": jnp.all(gap.law.within_validity),
        "corrosion": jnp.all(
            bundle.corrosion.validity.contains(
                interface_temperature=mean_outer_temperature
            )
        ),
        "hydrogen": jnp.all(
            bundle.hydrogen.validity.contains(temperature=mean_outer_temperature)
        ),
        "oxide_conductivity": jnp.all(
            bundle.oxide_conductivity.validity.contains(
                temperature=mean_outer_temperature
            )
        ),
        "bore_emissivity": jnp.all(
            (bore_temperature >= bore_low) & (bore_temperature <= bore_high)
        ),
        # The coolant-outlet-plus-offset plenum temperature has no source
        # validity range.
        "plenum_temperature": jnp.asarray(
            "plenum_temperature" not in STATICALLY_UNSOURCED_SLOTS
        ),
        "coolant": coolant_source_supported,
    }
    return jnp.concatenate(
        (
            fuel_support,
            cladding_support,
            jnp.stack(tuple(rod_support[slot] for slot in ROD_SOURCE_SLOTS)),
        )
    )


@final
class FuelStepLoading(StrictModule):
    """Step loading fixed by the accepted history, independent of the endpoint guess."""

    conditions: IrradiationConditions
    linear_power: Array
    fast_flux: Array
    radial_power: RadialPowerState
    radial_shape: Array
    fission_rate: Array
    produced_species: Array
    exposure: ExposureState


def evaluate_fuel_loading(
    prepared: FuelFormBinding,
    previous: FuelRodState,
    time: Array,
    dt: Array,
    bundle: FuelModelBundle,
    /,
) -> FuelStepLoading:
    """Step-average power and flux, burnup, radial power, and gas production.

    ``linear_power`` [W/m] and ``fast_flux`` [n/(m^2 s)] are step averages per
    zone; ``fission_rate`` [fissions/(m^3 s)] and ``produced_species``
    [atoms/(m^3 s)] are per fuel site.
    """
    geometry, specification = prepared.geometry, prepared.admission.specification
    history = prepared.admission.history
    fuel_mesh = geometry.fuel
    conditions = history.conditions(time + dt)
    linear_power = history.step_average_linear_power(time, time + dt)
    fast_flux = history.step_average_fast_flux(time, time + dt)
    fuel_area = jnp.pi * specification.pellet.radius**2
    material_fraction = 1.0 - specification.pellet.removed_volume_fraction
    burnup_increment = (
        linear_power
        * dt
        / (
            fuel_area
            * material_fraction
            * prepared.heavy_metal.heavy_metal_mass_density
            * JOULE_PER_MEGAWATT_DAY
        )
    )
    radial_power = bundle.radial_power.advance(
        previous.radial_power, burnup_increment_mwd_per_kghm=burnup_increment
    )
    radial_shape = bundle.radial_power.power_shape(radial_power).reshape(-1)
    fission_fractions = bundle.radial_power.fission_fractions(radial_power).reshape(
        (-1, len(FISSIONING_NUCLIDES))
    )
    energy_per_fission = fission_fractions @ bundle.yields.energy_per_fission
    recoverable_power_density = (
        linear_power[fuel_mesh.site_zone] * radial_shape / (fuel_area * material_fraction)
    )
    fission_rate_density = recoverable_power_density / energy_per_fission
    produced_species = jnp.zeros(
        (fuel_mesh.site_count, _GAS_SPECIES_COUNT), dtype=previous.fields.pressure.dtype
    )
    produced_species = produced_species.at[:, _KRYPTON].set(
        fission_rate_density * (fission_fractions @ bundle.yields.krypton_yields)
    )
    produced_species = produced_species.at[:, _XENON].set(
        fission_rate_density * (fission_fractions @ bundle.yields.xenon_yields)
    )
    helium_production = bundle.helium_source.production.evaluate(
        fission_rate_density,
        fission_fractions,
        bundle.helium_source.inventory,
        bundle.helium_source.reference_volume,
    )
    produced_species = produced_species.at[:, _HELIUM].set(helium_production.total)
    exposure = advance_exposure(
        previous.exposure,
        prepared.heavy_metal,
        fission_rate_density=fission_rate_density,
        energy_per_fission=energy_per_fission,
        fast_flux=fast_flux[fuel_mesh.site_zone],
        time_step=dt,
    )
    return FuelStepLoading(
        conditions=conditions,
        linear_power=linear_power,
        fast_flux=fast_flux,
        radial_power=radial_power,
        radial_shape=radial_shape,
        fission_rate=fission_rate_density,
        produced_species=produced_species,
        exposure=exposure,
    )


@eqx.filter_jit
def evaluate_fuel_gas(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    hydrostatic_stress: Array,
    previous: FuelRodState,
    dt: Array,
    bundle: FuelModelBundle,
    loading: FuelStepLoading,
    /,
) -> FuelGasBlockResult:
    """Advance the accepted gas, helium and tracer histories to this endpoint guess.

    Every call starts from the accepted histories of ``previous``; the free-gas
    helium partial pressure [Pa] is the grain-boundary helium condition.
    """
    fuel_mesh = prepared.geometry.fuel
    temperature = fuel_mesh.values(fields.temperature_fuel)
    previous_temperature = fuel_mesh.values(previous.fields.temperature_fuel)
    helium_pressure = (
        fields.pressure * fields.free_moles[_HELIUM] / jnp.sum(fields.free_moles)
    )

    def advance_site(
        gas: FissionGasSiteState,
        helium: HeliumGasSiteState,
        species: FissionGasSpeciesState,
        site_temperature: Array,
        previous_site_temperature: Array,
        burnup: Array,
        fission_rate_density: Array,
        species_production_rate: Array,
        site_hydrostatic_stress: Array,
    ) -> FuelGasBlockResult:
        return advance_fuel_gas(
            bundle.fission_gas,
            bundle.helium,
            gas,
            helium,
            species,
            temperature=site_temperature,
            previous_temperature=previous_site_temperature,
            burnup_mwd_per_kghm=burnup,
            fission_rate_density=fission_rate_density,
            species_production_rate=species_production_rate,
            hydrostatic_stress=site_hydrostatic_stress,
            time_step=dt,
            boundary_helium_pressure=helium_pressure,
        )

    return jax.vmap(advance_site)(
        previous.fuel_sites.gas.total,
        previous.fuel_sites.helium.state,
        previous.fuel_sites.gas.species,
        temperature,
        previous_temperature,
        loading.exposure.burnup_mwd_per_kghm,
        loading.fission_rate,
        loading.produced_species,
        hydrostatic_stress,
    )


def evaluate_fuel_forms(
    prepared: FuelFormBinding,
    fields: FuelRodFields,
    previous: FuelRodState,
    time: Array,
    dt: Array,
    bundle: FuelModelBundle,
    frozen_gas: FuelGasBlockResult,
    loading: FuelStepLoading,
    /,
) -> FuelFormEvaluation:
    """Residual rows, candidate state, and admissibility at one endpoint guess.

    Gas histories are frozen at ``frozen_gas`` and ``loading`` is fixed by the
    accepted history, so only the thermomechanical fields vary between calls.
    """
    geometry, specification = prepared.geometry, prepared.admission.specification
    fuel_interface = geometry.fuel_interface
    cladding_outer_surface = geometry.cladding_outer
    zone_count = geometry.layer_lengths.shape[0]
    material = _material_responses(
        prepared, fields, previous, dt, bundle, frozen_gas, loading
    )
    coolant_heat_flux = geometry.coolant_flux_values(fields.coolant_heat_flux)
    plenum = _plenum(
        prepared, fields, time, dt, bundle, loading, material, coolant_heat_flux
    )
    interface = _interface_kinematics(geometry, fields, plenum.response)
    gap = _gap_heat_transfer(prepared, fields, bundle, material, plenum, interface)
    gas = _gas_ledger(prepared, fields, previous, dt, loading, material.fuel)
    enthalpy = _stored_enthalpy(prepared, fields, bundle, material)
    fuel_thermal_rows, cladding_thermal_rows = _thermal_rows(
        prepared,
        fields,
        previous,
        dt,
        bundle,
        loading,
        material,
        enthalpy,
        gap,
        interface,
        coolant_heat_flux,
    )
    # Heat delivered to the coolant [W per zone] through the wall and directly.
    wall_power = jax.ops.segment_sum(
        cladding_outer_surface.weight * interface.outer_measure_ratio * coolant_heat_flux,
        cladding_outer_surface.zone,
        zone_count,
    )
    direct_coolant_power = (
        _direct_coolant_fraction(bundle) * loading.linear_power * geometry.layer_lengths
    )
    surface = _oxidation_surface(
        prepared,
        fields,
        previous,
        material.cladding.state,
        material.cladding_deformation,
        interface.outer_measure_ratio,
        coolant_heat_flux,
        loading.fast_flux,
        dt,
        bundle,
    )
    coolant = _coolant_rows(
        prepared,
        fields,
        previous,
        loading.conditions,
        time,
        dt,
        interface.outer_measure_ratio,
        coolant_heat_flux,
        surface.water_wall_temperature,
        wall_power,
        direct_coolant_power,
    )
    mechanical = _mechanical_rows(
        prepared,
        fields,
        previous,
        loading.conditions,
        plenum.response,
        material,
        surface.load_fraction,
        interface,
    )
    friction_rows = _friction_rows(prepared, fields, bundle, interface, mechanical.slip)
    contact_gap_rows = (
        zone_average(
            interface.contact.normal_gap,
            fuel_interface.weight,
            fuel_interface.interface_index,
            fields.contact_pressure.shape[0],
        )
        / specification.pellet.radius
    )
    fill_moles = jnp.sum(previous.gas.fill_moles)
    pressure_rows = gas_pressure_residual(
        fields.pressure, jnp.sum(fields.free_moles), plenum.response.components
    ) / (fill_moles * MOLAR_GAS_CONSTANT_J_PER_MOL_K)
    residual = FuelRodFields(
        temperature_fuel=fuel_thermal_rows / THERMAL_ROW_SCALE_W,
        temperature_cladding=cladding_thermal_rows / THERMAL_ROW_SCALE_W,
        displacement_fuel=mechanical.fuel
        / (
            MECHANICAL_ROW_SCALE_PA
            * specification.pellet.radius
            * specification.rod.stack_length
        ),
        displacement_cladding=mechanical.cladding
        / (
            MECHANICAL_ROW_SCALE_PA
            * specification.cladding.outer_radius
            * specification.rod.stack_length
        ),
        axial_strain_fuel=mechanical.fuel_axial,
        axial_strain_cladding=mechanical.cladding_axial,
        contact_pressure=contact_gap_rows,
        friction_force=friction_rows,
        pressure=pressure_rows,
        coolant_heat_flux=coolant.flux,
        free_moles=gas.balance_error / fill_moles,
        channel_state=coolant.channel,
    )
    energy = _energy_ledger(
        prepared,
        previous,
        dt,
        bundle,
        loading,
        enthalpy,
        wall_power,
        direct_coolant_power,
    )
    candidate = FuelRodState(
        time=time + dt,
        fields=fields,
        fuel_sites=material.fuel.state,
        cladding_sites=surface.state,
        exposure=loading.exposure,
        radial_power=loading.radial_power,
        gas=gas,
        energy=energy,
        prepared_id=prepared.prepared_id,
    )
    physical = _physically_admissible(
        prepared,
        fields,
        material,
        plenum.response,
        gas,
        gap,
        interface,
        surface,
        coolant.admissible,
    )
    source_support = _source_support(
        bundle, material, gap, surface.mean_temperature, coolant.source_supported
    )
    match bundle.applicability:
        case "strict-source-support":
            in_range = jnp.all(source_support)
        case "equation-reproduction":
            in_range = physical
        case unsupported:
            assert_never(unsupported)
    return FuelFormEvaluation(
        residual_fields=residual,
        candidate=candidate,
        local_converged=jnp.all(material.fuel.root_converged)
        & jnp.all(material.cladding.root_converged)
        & jnp.all(loading.radial_power.successful),
        in_range=in_range,
        physical=physical,
        contact_gap=interface.contact.normal_gap,
        friction_residual=friction_rows,
        gas_error=jnp.max(jnp.abs(gas.balance_error)),
        energy_error=energy.balance_error,
        accommodation_extrapolated=gap.law.accommodation_extrapolated,
        accommodation_source_comparison=gap.law.within_source_comparison,
        source_support=source_support,
    )

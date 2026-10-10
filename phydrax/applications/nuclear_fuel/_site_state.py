"""Accepted and candidate material histories at the prepared FE integration sites."""

from __future__ import annotations

from jax import Array

from ..._strict import StrictModule
from ._cladding_deformation import ZircaloyCreepState
from ._fuel_creep import FuelCreepState
from ._gas_state import (
    FissionGasDiagnostics,
    FissionGasSiteState,
    FissionGasSpeciesState,
    FissionGasSpeciesTransfers,
    FissionGasTransfers,
)
from ._helium import HeliumGasSiteState


class FuelSiteMechanics(StrictModule):
    """Creep history, eigenstrains and stresses of a fuel site.

    ``creep_log_strain`` is the accumulated logarithmic creep tensor;
    ``densification_strain`` and ``permanent_relocation_strain`` are linear
    strains; ``hydrostatic_stress`` and the 3x3 ``cauchy_stress`` are in Pa.
    """

    creep: FuelCreepState
    creep_log_strain: Array
    densification_strain: Array
    permanent_relocation_strain: Array
    hydrostatic_stress: Array
    cauchy_stress: Array


class FuelSiteGas(StrictModule):
    """Fission-gas totals, tracer species and their last-step transfers at a site.

    ``retained_species`` is in atoms per m^3 of reference fuel on the
    five-species gas axis in ``FUEL_GAS_SPECIES`` order; ``total`` is the
    total-gas state and ``species`` the passive tracer-species state.
    """

    total: FissionGasSiteState
    transfers: FissionGasTransfers
    diagnostics: FissionGasDiagnostics
    species: FissionGasSpeciesState
    species_transfers: FissionGasSpeciesTransfers
    retained_species: Array
    species_linear_status: Array
    species_linear_condition_estimate: Array


class FuelSiteHelium(StrictModule):
    """Helium shell state, last-step transfers and linear-solve status at a site.

    ``radial_cell_width`` is the grain-shell cell width in m; the condition
    estimate is zero where ``linear_conditioning_available`` is false.
    """

    state: HeliumGasSiteState
    released_increment: Array
    diffusion_to_boundary: Array
    grain_growth_sweeping: Array
    radial_cell_width: Array
    linear_status: Array
    linear_condition_estimate: Array
    linear_conditioning_available: Array


class FuelSiteState(StrictModule):
    """Accepted or candidate history of one fuel integration site.

    ``maximum_temperature`` is the largest site temperature reached, in K;
    ``constitutive_evaluated`` is false only for fabricated sites.
    """

    mechanics: FuelSiteMechanics
    gas: FuelSiteGas
    helium: FuelSiteHelium
    maximum_temperature: Array
    constitutive_evaluated: Array


class CladdingSiteState(StrictModule):
    creep: ZircaloyCreepState
    inelastic_log_strain: Array
    equivalent_plastic_strain: Array
    fast_fluence: Array
    oxide_thickness: Array
    hydrogen_ppm: Array
    maximum_temperature: Array
    hydrogen_areal_mass: Array
    cauchy_stress: Array
    plastic_active: Array
    constitutive_evaluated: Array

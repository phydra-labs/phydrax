#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Site dimensions, passive Kr/Xe identities and step records of the total-gas model.

Inventories are atoms per m^3 of reference fuel solid.
"""

from __future__ import annotations

from typing import final

import equinox as eqx

from ..._strict import StrictModule
from ...typing import (
    Bool,
    Dim,
    Float64,
    Int32,
    parse,
    VariadicDim,
)
from ._identity import FuelGasSpecies, IntragranularModeDim


class SiteDims(VariadicDim):
    """Arbitrary leading shape of fuel material sites."""


class FissionGasTracerSpeciesDim(Dim):
    """Passive Kr/Xe identities sharing one nonlinear gas geometry."""


FISSION_GAS_TRACER_SPECIES: tuple[FuelGasSpecies, ...] = (
    parse("Kr", FuelGasSpecies, "fission_gas_tracer_species"),
    parse("Xe", FuelGasSpecies, "fission_gas_tracer_species"),
)


@final
class FissionGasSiteState(StrictModule):
    """Accepted fission gas state of every fuel site (float64, leading shape ``S``).

    Inventories are atoms per m^3 of reference fuel solid. ``intragranular_modes`` are
    the modal coefficients of the equivalent-sphere concentration in the
    non-restructured grains; ``bubble_gas_atoms`` equals
    ``2 a boundary_gas / (3 N)`` for grain radius ``a`` and face bubble density
    ``N`` [1/m^2]. ``grain_face_intactness`` is the Barani (2017) fraction of
    intact grain faces, ``effective_burnup_mwd_per_kghm`` the Barani (2020)
    effective burnup and ``burnup_mwd_per_kghm`` the burnup of this state.
    ``hbs_pore_variance`` is integrated central second moment [atoms²/m³],
    ``hbs_pore_vacancies`` is total vacancy inventory [vacancies/m³].
    ``intragranular_unresolved`` is the retained Booth subgrid tail.
    """

    __strict_contract__ = True

    intragranular_modes: Float64[SiteDims, IntragranularModeDim]
    intragranular_unresolved: Float64[SiteDims]
    grain_radius: Float64[SiteDims]
    boundary_gas: Float64[SiteDims]
    bubble_number_density: Float64[SiteDims]
    bubble_vacancies: Float64[SiteDims]
    bubble_gas_atoms: Float64[SiteDims]
    grain_face_intactness: Float64[SiteDims]
    connected_boundary_volume: Float64[SiteDims]
    hbs_fraction: Float64[SiteDims]
    hbs_pore_gas: Float64[SiteDims]
    effective_burnup_mwd_per_kghm: Float64[SiteDims]
    hbs_boundary_gas: Float64[SiteDims]
    hbs_pore_number_density: Float64[SiteDims]
    hbs_pore_variance: Float64[SiteDims]
    hbs_pore_vacancies: Float64[SiteDims]
    hbs_porosity: Float64[SiteDims]
    burnup_mwd_per_kghm: Float64[SiteDims]
    produced: Float64[SiteDims]
    released: Float64[SiteDims]
    boundary_released: Float64[SiteDims]


@final
class FissionGasSpeciesState(StrictModule):
    """Passive Kr/Xe inventories, in that fixed order, of the total-gas reservoirs.

    Diffusion modes keep their separate identity mixtures. HBS pore gas and
    boundary gas are separate compartments with endpoint donor composition.
    Geometry and nonlinear moments are carried only by FissionGasSiteState.
    """

    __strict_contract__ = True
    intragranular_modes: Float64[
        SiteDims, IntragranularModeDim, FissionGasTracerSpeciesDim
    ]
    intragranular_unresolved: Float64[SiteDims, FissionGasTracerSpeciesDim]
    boundary_gas: Float64[SiteDims, FissionGasTracerSpeciesDim]
    hbs_pore_gas: Float64[SiteDims, FissionGasTracerSpeciesDim]
    hbs_boundary_gas: Float64[SiteDims, FissionGasTracerSpeciesDim]
    produced: Float64[SiteDims, FissionGasTracerSpeciesDim]
    released: Float64[SiteDims, FissionGasTracerSpeciesDim]
    boundary_released: Float64[SiteDims, FissionGasTracerSpeciesDim]
    species: tuple[FuelGasSpecies, ...] = eqx.field(
        static=True, default=FISSION_GAS_TRACER_SPECIES
    )


@final
class FissionGasSpeciesTransfers(StrictModule):
    """Per-identity transfers of one step; see FissionGasTransfers."""

    __strict_contract__ = True
    intragranular_production: Float64[SiteDims, FissionGasTracerSpeciesDim]
    hbs_production: Float64[SiteDims, FissionGasTracerSpeciesDim]
    athermal_release: Float64[SiteDims, FissionGasTracerSpeciesDim]
    grain_growth_sweeping: Float64[SiteDims, FissionGasTracerSpeciesDim]
    hbs_depletion: Float64[SiteDims, FissionGasTracerSpeciesDim]
    diffusion_to_boundary: Float64[SiteDims, FissionGasTracerSpeciesDim]
    burst_release: Float64[SiteDims, FissionGasTracerSpeciesDim]
    venting_release: Float64[SiteDims, FissionGasTracerSpeciesDim]
    hbs_nucleation: Float64[SiteDims, FissionGasTracerSpeciesDim]
    hbs_trapping: Float64[SiteDims, FissionGasTracerSpeciesDim]
    hbs_resolution: Float64[SiteDims, FissionGasTracerSpeciesDim]


@final
class FissionGasSpeciesStepResult(StrictModule):
    """Passive Kr/Xe candidate state, transfers and HBS exchange-solve status."""

    __strict_contract__ = True
    state: FissionGasSpeciesState
    transfers: FissionGasSpeciesTransfers
    released_increment: Float64[SiteDims, FissionGasTracerSpeciesDim]
    retained_total: Float64[SiteDims, FissionGasTracerSpeciesDim]
    linear_solve_successful: Bool[SiteDims]
    linear_solve_status: Int32[SiteDims]
    linear_solve_condition_estimate: Float64[SiteDims]
    linear_solve_residual_norm: Float64[SiteDims]
    admissible: Bool[SiteDims]
    in_range: Bool[SiteDims]


@final
class FissionGasTransfers(StrictModule):
    """Gas moved within one step [atoms/m^3 reference fuel], nonnegative when admitted.

    Production splits exactly into ``intragranular_production``,
    ``hbs_production`` and ``athermal_release``. Reservoir balances:
    intragranular ``+ intragranular_production - grain_growth_sweeping -
    hbs_depletion - diffusion_to_boundary``; grain faces ``+ grain_growth_sweeping
    + diffusion_to_boundary - burst_release - venting_release``; combined HBS
    storage ``+ hbs_production + hbs_depletion``. Inside HBS, pore gas changes
    by ``hbs_nucleation + hbs_trapping - hbs_resolution`` and its boundary
    inventory receives the complementary change. Re-solved HBS gas stays
    within HBS storage, never enters the rod free volume.
    """

    __strict_contract__ = True

    intragranular_production: Float64[SiteDims]
    hbs_production: Float64[SiteDims]
    athermal_release: Float64[SiteDims]
    grain_growth_sweeping: Float64[SiteDims]
    hbs_depletion: Float64[SiteDims]
    diffusion_to_boundary: Float64[SiteDims]
    burst_release: Float64[SiteDims]
    venting_release: Float64[SiteDims]
    hbs_nucleation: Float64[SiteDims]
    hbs_trapping: Float64[SiteDims]
    hbs_resolution: Float64[SiteDims]


@final
class FissionGasDiagnostics(StrictModule):
    """Per-step coverages, bubble pressure, truncation bound and root convergence."""

    __strict_contract__ = True

    effective_diffusivity: Float64[SiteDims]
    fractional_coverage: Float64[SiteDims]
    saturation_coverage: Float64[SiteDims]
    bubble_pressure: Float64[SiteDims]
    modal_truncation_bound: Float64[SiteDims]
    vacancy_root_converged: Bool[SiteDims]
    hbs_root_converged: Bool[SiteDims]
    hbs_root_residual_norm: Float64[SiteDims]
    hbs_root_condition_estimate: Float64[SiteDims]
    hbs_root_linear_status: Int32[SiteDims]
    hbs_root_linear_rank: Int32[SiteDims]
    hbs_root_conditioning_available: Bool[SiteDims]


@final
class FissionGasStepResult(StrictModule):
    """Candidate state, transfers, swelling strains and admissibility of one step.

    ``released_increment`` goes to the rod free volume. Swelling values are
    volumetric strains of the fuel solid: ``intragranular_swelling`` from the
    trapped intragranular gas, ``intergranular_swelling`` from closed grain-face
    bubbles (``(3 / (2a)) N V``) plus persisted connected cavities, and
    ``hbs_swelling`` from closed pore gas/vacancy volume. Only the connected
    cavity state contributes to rod free volume. ``admissible`` requires finite
    values, reconstructed concentration nonnegative on the 2N-point preparation
    grid, nonnegative inventories, coverage in ``[0,1]``, both roots converged
    and nondecreasing burnup. ``in_range`` reports the validity envelope.
    ``cumulative_fgr`` is boundary-only release divided by production, zero at
    zero production.
    """

    __strict_contract__ = True

    state: FissionGasSiteState
    transfers: FissionGasTransfers
    released_increment: Float64[SiteDims]
    retained_total: Float64[SiteDims]
    intragranular_swelling: Float64[SiteDims]
    intergranular_swelling: Float64[SiteDims]
    hbs_swelling: Float64[SiteDims]
    connected_boundary_volume_increment: Float64[SiteDims]
    cumulative_fgr: Float64[SiteDims]
    admissible: Bool[SiteDims]
    in_range: Bool[SiteDims]
    diagnostics: FissionGasDiagnostics


__all__ = [
    "FISSION_GAS_TRACER_SPECIES",
    "FissionGasDiagnostics",
    "FissionGasSiteState",
    "FissionGasSpeciesState",
    "FissionGasSpeciesStepResult",
    "FissionGasSpeciesTransfers",
    "FissionGasStepResult",
    "FissionGasTracerSpeciesDim",
    "FissionGasTransfers",
    "SiteDims",
]

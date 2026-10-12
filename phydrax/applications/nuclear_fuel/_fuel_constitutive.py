"""Local return mappings of the fuel and cladding material points.

Each call recomputes the endpoint response from the accepted site history and
returns an uncommitted candidate state.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import (
    Eigenproblem,
    ExponentialSpectralFunction,
    self_adjoint_spectral_operator,
    SelfAdjointSpectralOperatorPolicy,
)
from ...nonlinear import VectorLocalRootPlan
from ...operators.mechanics import (
    axisymmetric_hencky_stress,
    AxisymmetricHenckyStress,
    finite_strain_kinematics,
    FiniteStrainKinematics,
)
from ._bundles import FuelModelBundle
from ._cladding_deformation import ZircaloyCreepIncrement, ZircaloyPlasticityResult
from ._fuel_creep import FuelCreepUpdate
from ._fuel_eigenstrain import FuelDensificationUpdate
from ._fuel_properties import UO2_THEORETICAL_DENSITY, UO2ElasticModuli
from ._gas_block import FuelGasBlockResult
from ._gas_state import FissionGasSpeciesStepResult, FissionGasStepResult
from ._helium import HeliumGasStepResult
from ._layout import STATICALLY_UNSOURCED_SLOTS
from ._material_math import von_mises_equivalent
from ._numerical_views import FuelNumericalSpecification
from ._site_state import (
    CladdingSiteState,
    FuelSiteGas,
    FuelSiteHelium,
    FuelSiteMechanics,
    FuelSiteState,
)
from ._specification import FuelRodSpecification


_Specification: TypeAlias = FuelRodSpecification | FuelNumericalSpecification

_EXPONENTIAL = ExponentialSpectralFunction()
_SPECTRAL_POLICY = SelfAdjointSpectralOperatorPolicy(differentiation="frechet")
_SELF_ADJOINT = OperatorProperties(
    self_adjoint=True, evidence={"self_adjoint": "construction"}
)
_FUEL_CREEP_ROOT = VectorLocalRootPlan(
    1, maximum_steps=30, tolerance=1e-10, plan_id="fuel-creep-return"
)
_CLADDING_ROOT = VectorLocalRootPlan(
    4, maximum_steps=30, tolerance=1e-10, plan_id="sra-cladding-creep-plastic-return"
)


def multiplicative_hencky_stress(
    deformation: Array,
    eigen_log_strain: Array,
    bulk: Array,
    shear: Array,
    total_kinematics: FiniteStrainKinematics,
    /,
) -> AxisymmetricHenckyStress:
    """Hencky stress of the split ``F = Fe F*`` with ``F* = exp(eigen_log_strain)``.

    The eigenstrain may be anisotropic and carry shear, so ``F*`` need not
    commute with ``F``: the elastic part is ``Fe = F exp(-eigen_log_strain)``
    and the first Piola stress is pulled back through ``F*^-1``. The Jacobian
    and kinematic admissibility are those of the total deformation.
    """
    inverse_eigen = self_adjoint_spectral_operator(
        Eigenproblem(DenseLinearOperator(-eigen_log_strain, properties=_SELF_ADJOINT)),
        _EXPONENTIAL,
        policy=_SPECTRAL_POLICY,
    ).operator
    elastic = axisymmetric_hencky_stress(
        deformation @ inverse_eigen, jnp.zeros_like(eigen_log_strain), bulk, shear
    )
    return AxisymmetricHenckyStress(
        kirchhoff=elastic.kirchhoff,
        first_piola=elastic.first_piola @ inverse_eigen,
        elastic_log_strain=elastic.elastic_log_strain,
        jacobian=total_kinematics.jacobian,
        logarithm_status=elastic.logarithm_status,
        admissible=elastic.admissible
        & total_kinematics.admissible
        & jnp.all(jnp.isfinite(inverse_eigen)),
    )


def _deviatoric(tensor: Array, /) -> Array:
    return tensor - jnp.trace(tensor, axis1=-2, axis2=-1)[..., None, None] / 3 * jnp.eye(
        3, dtype=tensor.dtype
    )


class FuelConstitutiveResponse(StrictModule):
    state: FuelSiteState
    stress: AxisymmetricHenckyStress
    gas: FissionGasStepResult
    helium: HeliumGasStepResult
    species: FissionGasSpeciesStepResult
    conductivity_porosity: Array
    relocation_strain: Array
    thermal_relocation_strain: Array
    root_converged: Array
    in_range: Array
    source_support: Array


class _PreparedFuelSite(NamedTuple):
    """Eigenstrains and elastic moduli of a fuel site before the creep return."""

    as_fabricated_density: float | Array
    densification: FuelDensificationUpdate
    porosity: Array
    permanent_relocation: Array
    total_relocation: Array
    moduli: UO2ElasticModuli
    log_eigenstrain: Array


def _relocation_strains(
    bundle: FuelModelBundle,
    specification: _Specification,
    previous: FuelSiteState,
    burnup: Array,
    linear_power: Array,
    fission_rate_density: Array,
    /,
) -> tuple[Array, Array]:
    """Permanent (mechanical) and total (thermal and gas gap) relocation strains.

    Relocation starts with the first fission or produced gas. The permanent
    part never falls below its accepted value; the recoverable remainder is
    added only while the pellet is powered.
    """
    diametral_gap = 2 * specification.cold_gap
    pellet_diameter = 2 * specification.pellet.radius
    raw_permanent = bundle.relocation.mechanical_diametral_strain(
        linear_power, burnup, diametral_gap, pellet_diameter
    )
    raw_total = bundle.relocation.diametral_strain(
        linear_power, burnup, diametral_gap, pellet_diameter
    )
    activated = (fission_rate_density > 0) | (previous.gas.total.produced > 0)
    permanent = jnp.where(
        activated,
        jnp.maximum(previous.mechanics.permanent_relocation_strain, raw_permanent),
        0.0,
    )
    total = permanent + jnp.where(
        activated & (linear_power > 0),
        jnp.maximum(raw_total - raw_permanent, 0.0),
        0.0,
    )
    return permanent, total


def _prepare_fuel_site(
    bundle: FuelModelBundle,
    specification: _Specification,
    previous: FuelSiteState,
    temperature: Array,
    burnup: Array,
    linear_power: Array,
    fission_rate_density: Array,
    /,
) -> _PreparedFuelSite:
    pellet = specification.pellet
    as_fabricated_density = pellet.density_fraction * UO2_THEORETICAL_DENSITY
    densification = bundle.densification.update(
        burnup,
        pellet.resintering_density_change,
        as_fabricated_density,
        previous.mechanics.densification_strain,
    )
    porosity = 1.0 - pellet.density_fraction / (1.0 + densification.linear_strain) ** 3
    permanent_relocation, total_relocation = _relocation_strains(
        bundle, specification, previous, burnup, linear_power, fission_rate_density
    )
    thermal_strain = bundle.fuel_expansion.linear_strain(temperature)
    solid_swelling = bundle.solid_swelling.volumetric_strain(burnup)
    moduli = bundle.fuel_elasticity.moduli(temperature, porosity)
    isotropic_log_strain = (
        jnp.log1p(thermal_strain)
        + jnp.log1p(densification.linear_strain)
        + jnp.log1p(solid_swelling) / 3
    ) * jnp.eye(3, dtype=temperature.dtype)
    # Pellet fragments relocate radially and circumferentially, not axially.
    relocation_log_strain = jnp.diag(
        jnp.asarray(
            [jnp.log1p(permanent_relocation), jnp.log1p(permanent_relocation), 0.0]
        )
    )
    return _PreparedFuelSite(
        as_fabricated_density=as_fabricated_density,
        densification=densification,
        porosity=porosity,
        permanent_relocation=permanent_relocation,
        total_relocation=total_relocation,
        moduli=moduli,
        log_eigenstrain=isotropic_log_strain + relocation_log_strain,
    )


def _candidate_fuel_state(
    previous: FuelSiteState,
    frozen_gas: FuelGasBlockResult,
    creep: FuelCreepUpdate,
    stress: AxisymmetricHenckyStress,
    /,
    *,
    creep_log_strain: Array,
    densification_strain: Array,
    temperature: Array,
    hydrostatic_stress: Array,
    permanent_relocation_strain: Array,
) -> FuelSiteState:
    gas, helium, species = frozen_gas.gas, frozen_gas.helium, frozen_gas.species
    return FuelSiteState(
        mechanics=FuelSiteMechanics(
            creep=creep.state,
            creep_log_strain=creep_log_strain,
            densification_strain=densification_strain,
            permanent_relocation_strain=permanent_relocation_strain,
            hydrostatic_stress=hydrostatic_stress,
            cauchy_stress=stress.kirchhoff / stress.jacobian,
        ),
        gas=FuelSiteGas(
            total=gas.state,
            transfers=gas.transfers,
            diagnostics=gas.diagnostics,
            species=species.state,
            species_transfers=species.transfers,
            retained_species=frozen_gas.retained_species,
            species_linear_status=species.linear_solve_status,
            species_linear_condition_estimate=species.linear_solve_condition_estimate,
        ),
        helium=FuelSiteHelium(
            state=helium.state,
            released_increment=helium.released_increment,
            diffusion_to_boundary=helium.diffusion_to_boundary,
            grain_growth_sweeping=helium.grain_growth_sweeping,
            radial_cell_width=helium.radial_cell_width,
            linear_status=helium.linear_solve_status,
            linear_condition_estimate=jnp.where(
                helium.linear_solve_conditioning_available,
                helium.linear_solve_condition_estimate,
                0.0,
            ),
            linear_conditioning_available=helium.linear_solve_conditioning_available,
        ),
        maximum_temperature=jnp.maximum(previous.maximum_temperature, temperature),
        constitutive_evaluated=jnp.asarray(True),
    )


def _fuel_source_support(
    bundle: FuelModelBundle,
    specification: _Specification,
    prepared: _PreparedFuelSite,
    frozen_gas: FuelGasBlockResult,
    temperature: Array,
    rod_burnup: Array,
    /,
) -> Array:
    """Source-validity flags in the order of the fuel entries of ``source_ids``."""
    pellet = specification.pellet
    # GTRLOC specifies neither relocation activation nor retention on power
    # decrease, so the permanent-relocation history recurrence (the
    # ``relocation_history`` slot) has no source support.
    relocation_history_supported = jnp.asarray(
        "relocation_history" not in STATICALLY_UNSOURCED_SLOTS
    )
    return jnp.stack(
        (
            bundle.fuel_heat.validity.contains(temperature=temperature),
            bundle.fuel_expansion.validity.contains(temperature=temperature),
            bundle.fuel_conductivity.validity.contains(
                temperature=temperature,
                burnup_mwd_per_kghm=rod_burnup,
                as_fabricated_porosity=1 - pellet.density_fraction,
                gadolinia_weight_fraction=0.0,
            ),
            bundle.fuel_elasticity.validity.contains(
                temperature=temperature, porosity=prepared.porosity
            ),
            bundle.fuel_emissivity.validity.contains(temperature=temperature),
            bundle.densification.validity.contains(
                resintering_density_fraction=pellet.resintering_density_change
                / prepared.as_fabricated_density
            ),
            bundle.solid_swelling.validity.contains(burnup_mwd_per_kghm=rod_burnup),
            bundle.relocation.validity.contains(burnup_mwd_per_kghm=rod_burnup),
            bundle.fuel_creep.validity.contains(
                density_fraction=pellet.density_fraction,
                oxygen_to_metal_ratio=pellet.oxygen_to_metal_ratio,
            ),
            bundle.radial_power.validity.contains(burnup_mwd_per_kghm=rod_burnup),
            frozen_gas.gas.in_range,
            frozen_gas.helium.in_range,
            relocation_history_supported,
        )
    )


@eqx.filter_jit
def fuel_response(
    bundle: FuelModelBundle,
    specification: _Specification,
    previous: FuelSiteState,
    deformation: Array,
    temperature: Array,
    burnup: Array,
    rod_burnup: Array,
    linear_power: Array,
    fission_rate_density: Array,
    dt: Array,
    frozen_gas: FuelGasBlockResult,
    /,
) -> FuelConstitutiveResponse:
    """Fuel site return at the endpoint deformation and temperature.

    ``burnup`` is the local site burnup and ``rod_burnup`` the zone burnup in
    which the source validity ranges are stated [MWd/kgHM]. Gas swelling comes
    from the frozen endpoint gas block, which makes the hydrostatic relation
    explicit; only the deviatoric creep increment is a local root.
    """
    prepared = _prepare_fuel_site(
        bundle,
        specification,
        previous,
        temperature,
        burnup,
        linear_power,
        fission_rate_density,
    )
    bulk, shear = prepared.moduli.bulk, prepared.moduli.shear
    total_kinematics = finite_strain_kinematics(deformation)
    accepted_log_eigenstrain = (
        prepared.log_eigenstrain + previous.mechanics.creep_log_strain
    )
    trial = multiplicative_hencky_stress(
        deformation, accepted_log_eigenstrain, bulk, shear, total_kinematics
    )
    trial_deviatoric = _deviatoric(trial.kirchhoff / trial.jacobian)
    trial_equivalent = von_mises_equivalent(trial_deviatoric)
    # Prandtl-Reuss flow direction 3 s / (2 q) of the elastic trial stress.
    creep_direction = (
        1.5 * trial_deviatoric / jnp.where(trial_equivalent > 0, trial_equivalent, 1.0)
    )

    gas, helium = frozen_gas.gas, frozen_gas.helium
    gas_swelling = (
        gas.intragranular_swelling
        + gas.intergranular_swelling
        + gas.hbs_swelling
        + helium.intragranular_swelling
    )
    mechanical_log_volume = jnp.log(trial.jacobian) - jnp.trace(accepted_log_eigenstrain)
    # Everything except the creep increment is held fixed during the root.
    frozen_log_eigenstrain = (
        prepared.log_eigenstrain
        + jnp.log1p(gas_swelling) / 3 * jnp.eye(3, dtype=temperature.dtype)
        + previous.mechanics.creep_log_strain
    )

    def creep_residual(
        increment: Array,
    ) -> tuple[Array, tuple[AxisymmetricHenckyStress, FuelCreepUpdate]]:
        stress = multiplicative_hencky_stress(
            deformation,
            frozen_log_eigenstrain + increment[0] * creep_direction,
            bulk,
            shear,
            total_kinematics,
        )
        update = bundle.fuel_creep.update(
            von_mises_equivalent(_deviatoric(stress.kirchhoff / stress.jacobian)),
            temperature,
            fission_rate_density,
            specification.pellet.density_fraction,
            2 * gas.state.grain_radius,
            specification.pellet.oxygen_to_metal_ratio,
            previous.mechanics.creep,
            dt,
        )
        residual = jnp.reshape(
            increment[0] - update.equivalent_creep_strain_increment, (1,)
        )
        return residual, (stress, update)

    creep_root, creep_diagnostic, (stress, creep) = (
        _FUEL_CREEP_ROOT.solve_with_diagnostics(
            creep_residual, jnp.zeros((1,), dtype=temperature.dtype), has_aux=True
        )
    )
    creep_increment = creep_root[0]
    hydrostatic_stress = jnp.trace(stress.kirchhoff) / (3 * stress.jacobian)
    # Recheck the terminal evaluation: the creep increment must reproduce
    # itself and the explicit hydrostatic relation must hold.
    terminal_residual = jnp.stack(
        (
            creep_increment - creep.equivalent_creep_strain_increment,
            (mechanical_log_volume - jnp.log1p(gas_swelling)) / trial.jacobian
            - hydrostatic_stress / bulk,
        )
    )
    creep_converged = (
        creep_diagnostic.converged
        & stress.admissible
        & jnp.all(jnp.isfinite(terminal_residual))
        & (jnp.linalg.norm(terminal_residual) <= _FUEL_CREEP_ROOT.tolerance)
    )
    state = _candidate_fuel_state(
        previous,
        frozen_gas,
        creep,
        stress,
        creep_log_strain=previous.mechanics.creep_log_strain
        + creep_increment * creep_direction,
        densification_strain=prepared.densification.linear_strain,
        temperature=temperature,
        hydrostatic_stress=hydrostatic_stress,
        permanent_relocation_strain=prepared.permanent_relocation,
    )
    support = _fuel_source_support(
        bundle, specification, prepared, frozen_gas, temperature, rod_burnup
    )
    species = frozen_gas.species
    return FuelConstitutiveResponse(
        state=state,
        stress=stress,
        gas=gas,
        helium=helium,
        species=species,
        conductivity_porosity=prepared.porosity,
        relocation_strain=prepared.permanent_relocation,
        thermal_relocation_strain=prepared.total_relocation,
        root_converged=creep_converged
        & prepared.densification.root_converged
        & gas.diagnostics.vacancy_root_converged
        & gas.diagnostics.hbs_root_converged
        & species.linear_solve_successful
        & helium.linear_solve_successful,
        in_range=jnp.all(support),
        source_support=support,
    )


def _axisymmetric_tensor(components: Array, /) -> Array:
    """Symmetric ``(r, theta, z)`` tensor of ``(rr, theta theta, zz, rz)`` components."""
    return jnp.asarray(
        [
            [components[0], 0.0, components[3]],
            [0.0, components[1], 0.0],
            [components[3], 0.0, components[2]],
        ]
    )


def _axisymmetric_components(tensor: Array, /) -> Array:
    """Inverse of ``_axisymmetric_tensor`` for a symmetric axisymmetric tensor."""
    return jnp.stack((tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 2]))


class CladdingConstitutiveResponse(StrictModule):
    state: CladdingSiteState
    stress: AxisymmetricHenckyStress
    root_converged: Array
    in_range: Array
    source_support: Array


# Endpoint stress, its deviator, and the creep and plastic responses of the
# terminal root evaluation.
_CladdingTerminal: TypeAlias = tuple[
    AxisymmetricHenckyStress,
    Array,
    ZircaloyCreepIncrement,
    ZircaloyPlasticityResult,
]


def _cladding_source_support(
    bundle: FuelModelBundle,
    temperature: Array,
    fast_fluence: Array,
    cold_work: float | Array,
    equivalent_stress: Array,
    fast_flux: Array,
    loading_rate: Array,
    /,
) -> Array:
    """Source-validity flags in the order of the cladding entries of ``source_ids``."""
    return jnp.stack(
        (
            bundle.cladding_heat.validity.contains(temperature=temperature),
            bundle.cladding_conductivity.validity.contains(temperature=temperature),
            bundle.cladding_expansion.validity.contains(temperature=temperature),
            bundle.cladding_elasticity.validity.contains(
                temperature=temperature, cold_work=cold_work
            ),
            bundle.cladding_growth.validity.contains(fast_fluence=fast_fluence),
            bundle.cladding_creep.validity.contains(
                temperature=temperature,
                equivalent_stress=equivalent_stress,
                fast_flux=fast_flux,
            ),
            bundle.cladding_plasticity.validity.contains(
                temperature=temperature,
                fast_fluence=fast_fluence,
                cold_work=cold_work,
                equivalent_plastic_strain_rate=loading_rate,
            ),
        )
    )


@eqx.filter_jit
def cladding_response(
    bundle: FuelModelBundle,
    specification: _Specification,
    previous: CladdingSiteState,
    deformation: Array,
    previous_deformation: Array,
    temperature: Array,
    fast_flux: Array,
    dt: Array,
    /,
) -> CladdingConstitutiveResponse:
    """Cladding site return with coupled creep and plastic increments.

    The four local unknowns are the ``(rr, theta theta, zz, rz)`` components
    of the inelastic log-strain increment. Creep is evaluated at the endpoint
    stress and plasticity at the stress after the creep increment; the
    terminal root evaluation supplies the candidate creep history and plastic
    strain.
    """
    fast_fluence = previous.fast_fluence + dt * fast_flux
    cold_work = specification.cladding.cold_work
    thermal = bundle.cladding_expansion.principal_strains(temperature)
    growth = bundle.cladding_growth.principal_strains(fast_fluence)
    log_eigenstrain = jnp.diag(
        jnp.log1p(jnp.stack((thermal.radial, thermal.hoop, thermal.axial)))
        + jnp.log1p(jnp.stack((growth.radial, growth.hoop, growth.axial)))
    )
    accepted_log_eigenstrain = log_eigenstrain + previous.inelastic_log_strain
    moduli = bundle.cladding_elasticity.moduli(temperature, fast_fluence, cold_work, 0.0)
    bulk, shear = moduli.bulk, moduli.shear
    # Loading strain rate of the PNNL flow law, approximated by the von Mises
    # norm of the deviatoric deformation-gradient increment over the step.
    loading_rate = (
        von_mises_equivalent(_deviatoric(deformation - previous_deformation)) / dt
    )
    total_kinematics = finite_strain_kinematics(deformation)

    def residual(components: Array) -> tuple[Array, _CladdingTerminal]:
        stress = multiplicative_hencky_stress(
            deformation,
            accepted_log_eigenstrain + _axisymmetric_tensor(components),
            bulk,
            shear,
            total_kinematics,
        )
        deviatoric = _deviatoric(stress.kirchhoff / stress.jacobian)
        creep = bundle.cladding_creep.advance(
            deviatoric, temperature, fast_flux, fast_fluence, previous.creep, dt
        )
        after_creep = multiplicative_hencky_stress(
            deformation,
            accepted_log_eigenstrain + creep.strain_increment,
            bulk,
            shear,
            total_kinematics,
        )
        plastic = bundle.cladding_plasticity.return_map(
            _deviatoric(after_creep.kirchhoff / after_creep.jacobian),
            previous.equivalent_plastic_strain,
            loading_rate,
            temperature,
            fast_fluence,
            cold_work,
            dt,
            moduli=moduli,
        )
        inelastic = creep.strain_increment + plastic.plastic_strain_increment
        terminal = (stress, deviatoric, creep, plastic)
        return components - _axisymmetric_components(inelastic), terminal

    components, diagnostic, (stress, deviatoric, creep, plastic) = (
        _CLADDING_ROOT.solve_with_diagnostics(
            residual, jnp.zeros((4,), dtype=temperature.dtype), has_aux=True
        )
    )
    state = CladdingSiteState(
        creep=creep.state,
        inelastic_log_strain=previous.inelastic_log_strain
        + _axisymmetric_tensor(components),
        equivalent_plastic_strain=plastic.equivalent_plastic_strain,
        fast_fluence=fast_fluence,
        oxide_thickness=previous.oxide_thickness,
        hydrogen_ppm=previous.hydrogen_ppm,
        maximum_temperature=jnp.maximum(previous.maximum_temperature, temperature),
        hydrogen_areal_mass=previous.hydrogen_areal_mass,
        cauchy_stress=stress.kirchhoff / stress.jacobian,
        plastic_active=plastic.equivalent_plastic_strain
        > previous.equivalent_plastic_strain,
        constitutive_evaluated=jnp.asarray(True),
    )
    support = _cladding_source_support(
        bundle,
        temperature,
        fast_fluence,
        cold_work,
        von_mises_equivalent(deviatoric),
        fast_flux,
        loading_rate,
    )
    return CladdingConstitutiveResponse(
        state=state,
        stress=stress,
        root_converged=diagnostic.converged & plastic.converged,
        in_range=jnp.all(support),
        source_support=support,
    )

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._physical import RelativityScaleContract
from phydrax.applications.compact_objects._nonthermal_evolution import (
    NonthermalElectronEvolutionPlan,
    NonthermalLorentzGrid,
)
from phydrax.applications.compact_objects._plasma_closures import (
    TwoTemperatureElectronIonClosure,
)
from phydrax.applications.compact_objects._plasma_evolution import (
    ConstantElectronHeatingPlan,
    GyrotropicPlasmaClosurePlan,
    GyrotropicPlasmaState,
    PairCreationAnnihilationPlan,
    RelativisticPairState,
    RelativisticTwoTemperaturePlan,
)
from phydrax.applications.compact_objects._radiative_plasma import (
    GRPhotonNumberPlan,
    KleinNishinaScatteringPlan,
    ThermalBremsstrahlungGrayOpacityPlan,
    ThermalSynchrotronGrayOpacityPlan,
)
from phydrax.discretization.finite_volume._structured import FiniteVolumePlan
from phydrax.equations._relativistic_radiation_interaction import (
    ConstantGRGrayOpacityPlan,
)
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.solver._relativistic_finite_volume import (
    lower_valencia_stage_geometry,
)
from phydrax.units import KILOGRAM


def _scale():
    return RelativityScaleContract.geometric(KILOGRAM)


def _grid_stage(scale):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    discretization = FiniteVolumePlan(grid, component_names=("photon_number",)).prepare()
    convention = RelativityConvention.canonical()
    identity = jnp.broadcast_to(jnp.eye(3), (4, 3, 3))
    geometry = ADMGridGeometry(
        jnp.ones(4),
        jnp.zeros((4, 3)),
        identity,
        identity,
        jnp.ones(4),
        jnp.zeros((4, 3, 3)),
        jnp.ones(4, dtype="bool"),
        jnp.ones(4, dtype="bool"),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id=grid.topology.topology_id,
        geometry_lineage_id="flat",
    )
    return discretization, lower_valencia_stage_geometry(discretization, geometry, 0.0)


def test_physical_opacities_and_photon_number_sources_are_positive_and_balanced():
    scale = _scale()
    bremsstrahlung = ThermalBremsstrahlungGrayOpacityPlan(
        scale,
        emission_prefactor=2.0,
        minimum_temperature=0.1,
        maximum_temperature=10.0,
    ).evaluate(2.0, 1.0, 1.0, 0.0)
    synchrotron = ThermalSynchrotronGrayOpacityPlan(
        scale,
        electron_mass_per_particle=1.0,
        emission_prefactor=1.0,
        minimum_temperature=0.1,
        maximum_temperature=10.0,
    ).evaluate(2.0, 1.0, 1.0, 4.0)
    scattering_plan = KleinNishinaScatteringPlan(
        electron_mass_per_particle=1.0,
        thomson_cross_section=2.0,
        klein_nishina_temperature=1.0,
    )
    low_energy = scattering_plan.evaluate(1.0, 1.0, 0.1, 0.0)
    high_energy = scattering_plan.evaluate(1.0, 1.0, 10.0, 0.0)

    assert bool(bremsstrahlung.qualified)
    assert bool(synchrotron.qualified)
    assert float(bremsstrahlung.photon_emission_rate) > 0.0
    assert float(synchrotron.planck_emission) > 0.0
    assert float(low_energy.scattering) > float(high_energy.scattering)

    discretization, stage = _grid_stage(scale)
    photon_plan = GRPhotonNumberPlan(discretization, scale)
    state = photon_plan.initialize(jnp.ones(4), stage)
    opacity = ConstantGRGrayOpacityPlan(
        photon_absorption=1.0, photon_emission_rate=0.5
    ).evaluate(jnp.ones(4), jnp.ones(4), jnp.ones(4), jnp.zeros(4))
    radiation = jnp.broadcast_to(jnp.asarray((2.0, 0.0, 0.0, 0.0)), (4, 4))
    result = photon_plan.advance(state, radiation, opacity, stage, 0.1)

    assert bool(result.accepted)
    np.testing.assert_allclose(result.state.densitized_number, 1.05 / 1.1)
    np.testing.assert_allclose(result.ledger.balance_defect, 0.0, atol=1.0e-10)


def test_two_temperature_coulomb_exchange_conserves_total_species_energy():
    scale = _scale()
    coulomb = TwoTemperatureElectronIonClosure(
        scale,
        equilibration_time=1.0,
        minimum_temperature=0.1,
        maximum_temperature=10.0,
    )
    plan = RelativisticTwoTemperaturePlan(
        scale,
        ConstantElectronHeatingPlan(0.5),
        coulomb,
        electron_mass_per_particle=1.0,
        ion_mass_per_particle=1.0,
        electron_adiabatic_index=5.0 / 3.0,
        ion_adiabatic_index=5.0 / 3.0,
    )
    state = plan.initialize(jnp.ones(2), jnp.ones(2), 2.0 * jnp.ones(2))
    total = state.electron_internal_energy + state.ion_internal_energy
    result = plan.advance(
        state,
        jnp.ones(2),
        total,
        0.2 * jnp.ones(2),
        gas_pressure=jnp.ones(2),
        magnetic_pressure=jnp.ones(2),
    )

    assert bool(result.accepted)
    np.testing.assert_allclose(result.ledger.total_energy_defect, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        result.state.electron_internal_energy + result.state.ion_internal_energy,
        total,
    )
    assert bool(jnp.all(result.state.electron_temperature > state.electron_temperature))


def test_pair_reactions_preserve_charge_stoichiometry_and_total_energy():
    plan = PairCreationAnnihilationPlan(
        pair_rest_energy=1.0,
        creation_coefficient=0.2,
        annihilation_coefficient=0.01,
        threshold_temperature=1.0,
    )
    state = RelativisticPairState(
        jnp.asarray(1.0),
        jnp.asarray(0.5),
        jnp.asarray(10.0),
        jnp.asarray(5.0),
        jnp.asarray(20.0),
    )
    result = plan.advance(state, jnp.asarray(2.0), jnp.asarray(0.1))

    assert bool(result.accepted)
    assert float(result.ledger.pair_number_change) > 0.0
    np.testing.assert_allclose(result.ledger.charge_defect, 0.0)
    np.testing.assert_allclose(result.ledger.photon_stoichiometry_defect, 0.0)
    np.testing.assert_allclose(result.ledger.energy_defect, 0.0)


def test_nonthermal_injection_and_gyrotropic_conduction_close_their_ledgers():
    scale = _scale()
    grid = NonthermalLorentzGrid(jnp.asarray((1.0, 2.0, 4.0, 8.0)))
    nonthermal = NonthermalElectronEvolutionPlan(
        scale,
        grid,
        particle_rest_energy=1.0,
        injection_slope=2.5,
        injection_minimum=1.0,
        injection_maximum=8.0,
    )
    state = nonthermal.initialize(jnp.zeros((2, 3)))
    result = nonthermal.advance(
        state,
        jnp.asarray((0.01, 0.01)),
        expansion_rate=jnp.zeros(2),
        magnetic_squared=jnp.ones(2),
        radiation_energy_density=jnp.ones(2),
        radiation_temperature=jnp.ones(2),
        thermal_electron_density=jnp.ones(2),
        dissipative_heating=10.0 * jnp.ones(2),
        injection_fraction=0.5 * jnp.ones(2),
    )

    assert bool(result.accepted)
    np.testing.assert_allclose(result.ledger.energy_defect, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(result.ledger.number_defect, 0.0, atol=1.0e-8)
    assert bool(jnp.all(result.state.bin_number_density >= 0.0))

    gyrotropic = GyrotropicPlasmaClosurePlan(conductivity=2.0)
    gyro_state = GyrotropicPlasmaState(
        jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(2.0)
    )
    evaluation = gyrotropic.evaluate(
        gyro_state,
        jnp.asarray((2.0, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
        jnp.eye(3),
        jnp.eye(3),
    )
    np.testing.assert_allclose(evaluation.heat_flux, jnp.asarray((-2.0, 0.0, 0.0)))
    assert float(evaluation.entropy_production) > 0.0
    assert bool(evaluation.qualified)

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy import integrate, special

import phydrax as phx
from phydrax._physical import ElectromagneticScaleContract, RelativityScaleContract
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


def _scale() -> Any:
    return RelativityScaleContract.geometric(KILOGRAM)


def _grid_stage(scale: Any) -> Any:
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


def test_compact_object_plasma_evolution_scenario_1() -> None:
    scale = _scale()
    scattering_plan = KleinNishinaScatteringPlan(
        electron_mass_per_particle=1.0,
        thomson_cross_section=2.0,
        klein_nishina_temperature=1.0,
    )
    low_energy = scattering_plan.evaluate(1.0, 1.0, 0.1, 0.0)
    high_energy = scattering_plan.evaluate(1.0, 1.0, 10.0, 0.0)

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


def test_nonthermal_injection_and_gyrotropic_conduction_close_their_ledgers() -> None:
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


# CODATA 2022 SI constants, written independently of the scale contract.
_E = 1.602176634e-19
_ME = 9.1093837139e-31
_C = 299792458.0
_EPS0 = 8.8541878188e-12
_KB = 1.380649e-23
_H = 6.62607015e-34
_STEFAN_OVER_PI = 2.0 * np.pi**4 * _KB**4 / (15.0 * _H**3 * _C**2)


def _free_free_emissivity_per_hertz(
    electrons: float, ions: float, temperature: float, u: float
) -> float:
    """Rybicki & Lightman eq. 5.14a per steradian with the Born thermal Gaunt factor."""
    prefactor = (
        32.0 * np.pi * _E**6 / (3.0 * _ME * _C**3 * (4.0 * np.pi * _EPS0) ** 3)
    ) * np.sqrt(2.0 * np.pi / (3.0 * _KB * _ME))
    gaunt = np.sqrt(3.0) / np.pi * special.kve(np.float64(0.0), np.float64(0.5 * u))
    return (
        prefactor
        * electrons
        * ions
        * np.exp(-u)
        * gaunt
        / np.sqrt(temperature)
        / (4.0 * np.pi)
    )


def test_free_free_gray_closure_is_the_born_planck_and_rosseland_mean() -> None:
    scale = ElectromagneticScaleContract.si()
    proton_mass = 1.67262192595e-27
    density, temperature, radiation = 1.0e-3, 1.0e8, 5.0e7
    evaluation = ThermalBremsstrahlungGrayOpacityPlan(
        scale, electron_mass_per_particle=proton_mass
    ).evaluate(density, temperature, radiation, 0.0)
    electrons = density / proton_mass
    # The Planck average of the Born thermal Gaunt factor is exactly 2√3/π.
    prefactor = (
        32.0 * np.pi * _E**6 / (3.0 * _ME * _C**3 * (4.0 * np.pi * _EPS0) ** 3)
    ) * np.sqrt(2.0 * np.pi / (3.0 * _KB * _ME))
    total = (
        prefactor
        * electrons**2
        / np.sqrt(temperature)
        * (_KB * temperature / _H)
        * 2.0
        * np.sqrt(3.0)
        / np.pi
        / (4.0 * np.pi)
    )
    np.testing.assert_allclose(
        evaluation.planck_emission,
        total / (_STEFAN_OVER_PI * temperature**4),
        rtol=1.0e-6,
    )

    def alpha(nu: float) -> float:
        u = _H * nu / (_KB * temperature)
        j = _free_free_emissivity_per_hertz(electrons, electrons, temperature, u)
        return float(j * _C**2 * np.expm1(u) / (2.0 * _H * nu**3))

    def planck(nu: float, t: float) -> float:
        return float(2.0 * _H * nu**3 / _C**2 / np.expm1(_H * nu / (_KB * t)))

    def log_integral(function: Callable[[float], float]) -> float:
        lower, upper = (
            np.log(1.0e-9 * _KB * radiation / _H),
            np.log(80.0 * _KB * temperature / _H),
        )
        return integrate.quad(
            lambda s: function(np.exp(s)) * np.exp(s), lower, upper, limit=500
        )[0]

    absorption = log_integral(lambda nu: alpha(nu) * planck(nu, radiation))
    np.testing.assert_allclose(
        evaluation.planck_absorption,
        absorption / (_STEFAN_OVER_PI * radiation**4),
        rtol=1.0e-6,
    )

    def rosseland_weight(nu: float) -> float:
        u = _H * nu / (_KB * temperature)
        return planck(nu, temperature) * u / -np.expm1(-u) / temperature

    inverse = log_integral(lambda nu: rosseland_weight(nu) / alpha(nu))
    np.testing.assert_allclose(
        evaluation.rosseland_transport,
        4.0 * _STEFAN_OVER_PI * temperature**3 / inverse,
        rtol=1.0e-6,
    )
    assert bool(evaluation.qualified)
    radiation_constant = 4.0 * np.pi * _STEFAN_OVER_PI / _C
    np.testing.assert_allclose(
        evaluation.photon_emission_rate,
        evaluation.planck_emission
        * radiation_constant
        * temperature**3
        / (2.70118 * _KB),
        rtol=1.0e-8,
    )

    # Below the Born support (Z² Ry ≳ kT) the closure is unqualified.
    cold = ThermalBremsstrahlungGrayOpacityPlan(
        scale, electron_mass_per_particle=proton_mass
    ).evaluate(density, 1.0e4, 1.0e4, 0.0)
    assert not bool(cold.qualified)


def test_synchrotron_gray_closure_uses_the_mny96_planck_mean() -> None:
    scale = ElectromagneticScaleContract.si()
    proton_mass = 1.67262192595e-27
    density, temperature, radiation, field = 1.0e-14, 1.0e11, 5.0e10, 1.0e-1
    permeability = 1.0 / (_EPS0 * _C**2)
    evaluation = ThermalSynchrotronGrayOpacityPlan(
        scale, electron_mass_per_particle=proton_mass
    ).evaluate(density, temperature, radiation, field**2 / permeability)
    electrons = density / proton_mass
    theta = _KB * temperature / (_ME * _C**2)
    nu_s = 1.5 * _E * field / (2.0 * np.pi * _ME) * theta**2

    def emissivity(log_x: float) -> float:
        x = np.exp(log_x)
        nu = x * nu_s
        shape = (
            4.0505
            * x ** (-1.0 / 6.0)
            * (1.0 + 0.40 * x ** (-0.25) + 0.5316 * x ** (-0.5))
            * np.exp(-1.8899 * x ** (1.0 / 3.0))
        )
        j = (
            electrons
            * _E**2
            * nu
            / (4.0 * np.pi * _EPS0 * _C * np.sqrt(3.0) * special.kn(2, 1.0 / theta))
            * shape
        )
        return float(j * nu)

    total = integrate.quad(emissivity, np.log(1.0e-6), np.log(1.0e6), limit=400)[0]
    np.testing.assert_allclose(
        evaluation.planck_emission,
        total / (_STEFAN_OVER_PI * temperature**4),
        rtol=1.0e-6,
    )
    assert bool(evaluation.physically_valid)
    # The Rosseland weight lies far above the MNY96 frequency support.
    assert not bool(evaluation.qualified)

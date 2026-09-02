#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.reacting_flow._cantera import (
    CanteraNonDifferentiableBoundaryError,
    CanteraReferenceAdapter,
    CanteraUnsupportedFeatureError,
    CanteraYAMLAdapter,
)
from phydrax.applications.reacting_flow._low_mach import LowMachReactingFormulation
from phydrax.applications.reacting_flow._mechanism import ChemicalMechanismCompiler
from phydrax.applications.reacting_flow._state import ReactiveConservedLayout
from phydrax.applications.reacting_flow._statistics import (
    ReactiveClosureTargetPlan,
    ReactiveFlowStatisticsPlan,
)
from phydrax.applications.reacting_flow._thermodynamics import ReactingGasModel
from phydrax.equations._chemical_mechanism import (
    ChemicalMechanismIR,
    ChemicalReactionSpec,
)
from phydrax.equations._chemical_rates import (
    ArrheniusRatePlan,
    ChebyshevRatePlan,
    LindemannRatePlan,
    PLogRatePlan,
    ThirdBodyRatePlan,
    TroeRatePlan,
)
from phydrax.equations._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from phydrax.equations._chemical_thermodynamics import (
    PolynomialSpeciesThermodynamicsPlan,
)
from phydrax.solver._chemical_reactor import ChemicalReactorKind, ChemicalReactorPlan


def _mechanism():
    schema = ChemicalSpeciesSchema(
        ("A", "B"),
        (ChemicalPhaseKind.GAS, ChemicalPhaseKind.GAS),
        jnp.asarray((0.01, 0.01)),
        ("E",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.asarray((1, 1), dtype=jnp.int32),
    )
    thermo = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 20.0)),
        jnp.asarray((0.0, 0.0)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=3000.0,
    )
    elementary = ArrheniusRatePlan(2.0, 0.1, 1000.0)
    low = ArrheniusRatePlan(1.0e-2)
    high = ArrheniusRatePlan(3.0)
    efficiencies = jnp.asarray((1.0, 2.0))
    rates = (
        elementary,
        ThirdBodyRatePlan(ArrheniusRatePlan(1.0), efficiencies),
        LindemannRatePlan(low, high, efficiencies),
        TroeRatePlan(low, high, efficiencies, 0.5, 1000.0, 5000.0, 100.0),
        PLogRatePlan(
            jnp.asarray((1.0e4, 1.0e6)),
            (ArrheniusRatePlan(1.0), ArrheniusRatePlan(4.0)),
        ),
        ChebyshevRatePlan(
            jnp.asarray(((0.0, 0.1), (0.2, 0.0))),
            300.0,
            2000.0,
            1.0e4,
            1.0e6,
        ),
    )
    reactions = tuple(
        ChemicalReactionSpec(
            f"A->B:{index}",
            {"A": 1.0},
            {"B": 1.0},
            rate,
        )
        for index, rate in enumerate(rates)
    )
    ir = ChemicalMechanismIR("all-gas-features", schema, thermo, reactions)
    gas = ReactingGasModel(
        schema,
        thermo,
        formation_molar_enthalpies=jnp.asarray((0.0, -5.0e4)),
    )
    return gas, ChemicalMechanismCompiler().compile(ir, gas_model=gas)


def test_compiler_reports_arrhenius_third_body_falloff_and_pressure_dependence():
    _, mechanism = _mechanism()
    counts = dict(mechanism.features.rate_kind_counts)

    assert counts == {
        "arrhenius": 1,
        "third_body": 1,
        "lindemann": 1,
        "troe": 1,
        "plog": 1,
        "chebyshev": 1,
    }
    assert mechanism.features.third_body_reaction_count == 1
    assert mechanism.features.falloff_reaction_count == 2
    assert mechanism.features.pressure_dependent_reaction_count == 2


def test_compiled_sources_preserve_element_charge_mass_and_energy_bookkeeping():
    gas, mechanism = _mechanism()
    result = mechanism.evaluate(
        jnp.asarray((2.0, 1.0)),
        jnp.asarray(900.0),
        jnp.asarray(101325.0),
    )

    assert result.evidence.successful
    np.testing.assert_allclose(result.evidence.element_residual, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.evidence.charge_residual, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.evidence.energy_residual, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        jnp.sum(result.species_mass_production_rate), 0.0, atol=1.0e-12
    )
    expected_heat = -jnp.sum(
        result.species_molar_production_rate
        * (
            gas.thermodynamics.evaluate(jnp.asarray(900.0)).molar_enthalpy
            + gas.formation_molar_enthalpies
        )
    )
    np.testing.assert_allclose(result.heat_release_rate, expected_heat)


def test_constant_volume_and_pressure_reactor_states_use_correct_constraints():
    _, mechanism = _mechanism()
    amounts = jnp.asarray((2.0, 1.0))
    constant_volume = ChemicalReactorPlan(
        mechanism.prepared,
        ChemicalReactorKind.ADIABATIC_CONSTANT_VOLUME,
        fixed_volume=1.0,
    )
    volume_state = constant_volume.initial_state(amounts, jnp.asarray(800.0))
    volume_evaluation = constant_volume.evaluate(volume_state)
    constant_pressure = ChemicalReactorPlan(
        mechanism.prepared,
        ChemicalReactorKind.ISOTHERMAL_CONSTANT_PRESSURE,
        fixed_temperature=800.0,
        fixed_pressure=101325.0,
    )
    pressure_evaluation = constant_pressure.evaluate(amounts)

    assert volume_evaluation.successful
    assert pressure_evaluation.successful
    np.testing.assert_allclose(volume_evaluation.volume, 1.0)
    np.testing.assert_allclose(pressure_evaluation.pressure, 101325.0)
    np.testing.assert_allclose(pressure_evaluation.temperature, 800.0)


def test_low_mach_divergence_source_keeps_thermodynamic_pressure_separate():
    gas, mechanism = _mechanism()
    formulation = LowMachReactingFormulation(gas, 2, mechanism=mechanism)
    mass = jnp.asarray((0.3, 0.7))
    mass_rate = jnp.asarray((-0.02, 0.02))
    evidence = formulation.divergence_source(
        jnp.asarray(1000.0),
        mass,
        jnp.asarray(20.0),
        mass_rate,
        jnp.asarray(2.0e5),
        thermodynamic_pressure_rate=jnp.asarray(1000.0),
    )

    expected = 20.0 / 1000.0 - 1000.0 / 2.0e5
    assert evidence.successful
    np.testing.assert_allclose(evidence.compositional_expansion, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(evidence.divergence_source, expected)


def test_reactive_favre_species_element_energy_and_closure_statistics():
    gas, _ = _mechanism()
    layout = ReactiveConservedLayout(gas, 1)
    first = layout.from_thermodynamic_state(
        jnp.asarray(1.0), jnp.asarray((2.0,)), jnp.asarray(700.0), jnp.asarray((0.8, 0.2))
    )
    second = layout.from_thermodynamic_state(
        jnp.asarray(2.0),
        jnp.asarray((-1.0,)),
        jnp.asarray(900.0),
        jnp.asarray((0.2, 0.8)),
    )
    conserved = jnp.stack((first, second))
    targets = ReactiveClosureTargetPlan(layout).build(
        jnp.asarray(((-1.0, 1.0), (-2.0, 2.0))),
        jnp.asarray((3.0, 5.0)),
        jnp.asarray((((1.0,), (-1.0,)), ((2.0,), (-2.0,)))),
        jnp.asarray(((0.5,), (1.5,))),
        jnp.asarray((0.2, 0.4)),
    )
    statistics = ReactiveFlowStatisticsPlan(layout).evaluate(
        conserved,
        jnp.asarray((1.0, 1.0)),
        closure_targets=targets,
    )

    assert statistics.successful
    np.testing.assert_allclose(statistics.mean_density, 1.5)
    np.testing.assert_allclose(statistics.favre_velocity, 0.0)
    np.testing.assert_allclose(statistics.favre_species_mass_fractions, (0.4, 0.6))
    np.testing.assert_allclose(statistics.mean_element_amount_per_mass, 100.0)
    np.testing.assert_allclose(statistics.mean_heat_release_rate, 4.0)
    assert jnp.all(jnp.linalg.eigvalsh(statistics.favre_species_covariance) >= -1.0e-12)


def test_cantera_adapter_reports_unsupported_features_and_refuses_device_values(tmp_path):
    source = tmp_path / "surface.yaml"
    source.write_text(
        """phases:
- name: gas
  thermo: ideal-surface
  species: all
species:
- name: A
  composition: {H: 1}
  thermo: {model: constant-cp}
reactions: []
""",
        encoding="utf-8",
    )
    adapter = CanteraYAMLAdapter("gas")
    report = adapter.inspect(source)

    assert not report.supported
    assert not report.differentiable
    with pytest.raises(CanteraUnsupportedFeatureError):
        adapter.import_mechanism(source)
    reference = CanteraReferenceAdapter(object(), solution_id="unreached")
    with pytest.raises(CanteraNonDifferentiableBoundaryError):
        reference.evaluate(jnp.asarray(300.0), 101325.0, np.asarray((1.0,)))


def test_cantera_host_yaml_import_and_reference_state_are_explicit(tmp_path):
    source = tmp_path / "gas.yaml"
    source.write_text(
        """description: supported-gas
units:
  length: m
  quantity: mol
  activation-energy: J/mol
  pressure: Pa
phases:
- name: gas
  thermo: ideal-gas
  kinetics: gas
  transport: mixture-averaged
  species: [H, Hx]
species:
- name: H
  composition: {H: 1}
  thermo:
    model: NASA7
    temperature-ranges: [200.0, 3000.0]
    data: [[3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
- name: Hx
  composition: {H: 1}
  thermo:
    model: NASA7
    temperature-ranges: [200.0, 3000.0]
    data: [[3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
reactions:
- equation: H => Hx
  rate-constant: {A: 2.0, b: 0.0, Ea: 0.0}
""",
        encoding="utf-8",
    )
    imported = CanteraYAMLAdapter("gas").import_mechanism(source)
    assert imported.report.supported
    assert not imported.report.differentiable
    assert imported.mechanism.prepare().reaction_count == 1

    class _ReferenceSolution:
        X = np.asarray((0.25, 0.75))
        net_production_rates = np.asarray((-2.0, 2.0))
        density = 0.4
        mean_molecular_weight = 10.0
        cp_mass = 1200.0
        cv_mass = 900.0
        enthalpy_mass = 3.0e5
        int_energy_mass = 2.0e5
        heat_release_rate = 4.0e4
        TPY = None

    reference = CanteraReferenceAdapter(
        _ReferenceSolution(), solution_id="fixed-reference"
    ).evaluate(800.0, 101325.0, np.asarray((0.25, 0.75)))
    assert reference.temperature == 800.0
    assert reference.density == 0.4
    assert reference.mean_molar_mass == 0.01
    assert reference.species_molar_production_rate == (-2.0, 2.0)

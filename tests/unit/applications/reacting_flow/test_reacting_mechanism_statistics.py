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
from phydrax.applications.reacting_flow._statistics import (
    ReactiveClosureTargetPlan,
    ReactiveFlowStatisticsPlan,
)
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
from phydrax.equations._gas_dynamics import HomogeneousMixtureEulerSystem
from phydrax.equations._homogeneous_thermodynamics import (
    HomogeneousHelmholtzPlan,
    IdealGasReferenceHelmholtzTerm,
    ZeroResidualHelmholtzTerm,
)
from phydrax.solver._chemical_reactor import ChemicalReactorKind, ChemicalReactorPlan


def _mechanism():
    schema = ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (ChemicalPhaseKind.GAS, ChemicalPhaseKind.GAS),
        jnp.asarray((0.01, 0.01)),
        ("E",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.asarray((1, 1), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
        provenance="reacting-mechanism-test",
    )
    species_thermodynamics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 20.0)),
        jnp.asarray((0.0, -5.0e4)),
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
    prepared = ChemicalMechanismIR(
        "all-gas-features", schema, species_thermodynamics, reactions
    ).prepare()
    thermodynamics = HomogeneousHelmholtzPlan(
        IdealGasReferenceHelmholtzTerm(schema, species_thermodynamics),
        ZeroResidualHelmholtzTerm(schema),
    )
    return thermodynamics, prepared, HomogeneousMixtureEulerSystem(thermodynamics, 1)


def test_prepared_mechanism_owns_all_canonical_rate_plans_and_exact_schema():
    thermodynamics, mechanism, _ = _mechanism()
    kinds = tuple(reaction.forward_rate.kind.value for reaction in mechanism.reactions)

    assert kinds == (
        "arrhenius",
        "third_body",
        "lindemann",
        "troe",
        "plog",
        "chebyshev",
    )
    assert mechanism.reaction_count == 6
    assert mechanism.preparation_evidence.balanced
    assert mechanism.schema.schema_id == thermodynamics.schema.schema_id
    assert (
        mechanism.thermodynamics.thermodynamics_id
        == thermodynamics.thermodynamics.thermodynamics_id
    )


def test_canonical_sources_preserve_element_charge_mass_and_total_energy():
    _, mechanism, system = _mechanism()
    result = mechanism.evaluate(
        jnp.asarray((2.0, 1.0)),
        jnp.asarray(900.0),
        jnp.asarray(101325.0),
    )
    mass_rate = result.species_amount_rate * mechanism.schema.molar_masses
    conservative_source = (
        jnp.zeros((system.component_count,)).at[: system.species_count].set(mass_rate)
    )
    heat_release = -jnp.sum(
        result.species_amount_rate * result.thermodynamics.molar_enthalpy
    )

    assert result.successful
    np.testing.assert_allclose(result.element_residual, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.charge_residual, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(jnp.sum(mass_rate), 0.0, atol=1.0e-12)
    np.testing.assert_array_equal(conservative_source[-1], 0.0)
    assert heat_release > 0.0


def test_constant_volume_and_pressure_reactors_consume_prepared_mechanism():
    _, mechanism, _ = _mechanism()
    amounts = jnp.asarray((2.0, 1.0))
    constant_volume = ChemicalReactorPlan(
        mechanism,
        ChemicalReactorKind.ADIABATIC_CONSTANT_VOLUME,
        fixed_volume=1.0,
    )
    volume_state = constant_volume.initial_state(amounts, jnp.asarray(800.0))
    volume_evaluation = constant_volume.evaluate(volume_state)
    constant_pressure = ChemicalReactorPlan(
        mechanism,
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


def test_low_mach_uses_full_species_and_canonical_thermodynamic_derivatives():
    thermodynamics, mechanism, _ = _mechanism()
    formulation = LowMachReactingFormulation(thermodynamics, 2, mechanism=mechanism)
    mass = jnp.asarray((0.3, 0.7))
    mass_rate = jnp.asarray((-0.02, 0.02))
    temperature = jnp.asarray(1000.0)
    pressure = jnp.asarray(2.0e5)
    pressure_rate = jnp.asarray(1000.0)
    evidence = formulation.divergence_source(
        temperature,
        mass,
        jnp.asarray(20.0),
        mass_rate,
        pressure,
        thermodynamic_pressure_rate=pressure_rate,
    )
    state = formulation.initial_state(
        jnp.asarray((0.0, 0.0)), temperature, mass, pressure
    )
    chemistry = formulation.evaluate_chemistry(state)
    expected = (
        evidence.thermal_expansion
        + evidence.compositional_expansion
        + evidence.pressure_expansion
    )

    assert state.mass_fractions.shape == (mechanism.schema.species_count,)
    assert evidence.successful
    assert chemistry.divergence.successful
    assert chemistry.successful
    np.testing.assert_allclose(evidence.compositional_expansion, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(evidence.divergence_source, expected)
    np.testing.assert_allclose(evidence.thermal_expansion, 20.0 / 1000.0)
    np.testing.assert_allclose(evidence.pressure_expansion, -1000.0 / 2.0e5)
    np.testing.assert_allclose(
        jnp.sum(chemistry.species_mass_production_rate), 0.0, atol=1.0e-12
    )
    assert chemistry.diagnostic_heat_release_rate > 0.0


def test_reactive_favre_species_element_energy_and_closure_statistics():
    _, _, system = _mechanism()
    first = system.primitive_to_conserved(jnp.asarray((0.8, 0.2, 2.0, 700.0)))
    second = system.primitive_to_conserved(jnp.asarray((0.4, 1.6, -1.0, 900.0)))
    conserved = jnp.stack((first, second))
    targets = ReactiveClosureTargetPlan(system).build(
        jnp.asarray(((-1.0, 1.0), (-2.0, 2.0))),
        jnp.asarray((3.0, 5.0)),
        jnp.asarray((((1.0,), (-1.0,)), ((2.0,), (-2.0,)))),
        jnp.asarray(((0.5,), (1.5,))),
        jnp.asarray((0.2, 0.4)),
    )
    statistics = ReactiveFlowStatisticsPlan(system).evaluate(
        conserved,
        jnp.asarray((1.0, 1.0)),
        closure_targets=targets,
    )

    assert targets.successful.all()
    np.testing.assert_array_equal(targets.energy_source, 0.0)
    np.testing.assert_allclose(targets.element_source, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(targets.charge_source, 0.0, atol=1.0e-12)
    assert statistics.successful
    np.testing.assert_allclose(statistics.mean_density, 1.5)
    np.testing.assert_allclose(statistics.favre_velocity, 0.0)
    np.testing.assert_allclose(statistics.favre_species_mass_fractions, (0.4, 0.6))
    np.testing.assert_allclose(statistics.mean_element_amount_per_mass, 100.0)
    np.testing.assert_allclose(statistics.mean_diagnostic_heat_release_rate, 4.0)
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


def test_cantera_host_yaml_builds_catalog_gas_schema_thermo_and_mechanism(tmp_path):
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
    assert imported.catalog.catalog_id == imported.schema.catalog.catalog_id
    assert imported.schema.phase_count == 1
    assert imported.schema.phase_specs[0].kind is ChemicalPhaseKind.GAS
    assert imported.schema.phase_specs[0].standard_pressure == 101325.0
    assert imported.thermodynamics.schema.schema_id == imported.schema.schema_id
    assert imported.mechanism.schema.schema_id == imported.schema.schema_id
    assert imported.mechanism.reaction_count == 1

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
    assert reference.diagnostic_heat_release_rate == 4.0e4

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.compressible_flow._qualification import (
    ManufacturedViscousNSPlan,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._conservation_boundary import ExtrapolationBoundary
from phydrax.discretization.fem._boundary import FiniteElementBoundarySet
from phydrax.discretization.fem._generic import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization.fem._reference import discontinuous_element
from phydrax.discretization.finite_volume._riemann import RusanovFluxPlan
from phydrax.equations._chemical_species import (
    ChemicalPhaseKind,
    ChemicalSpeciesSchema,
)
from phydrax.equations._chemical_thermodynamics import (
    PolynomialSpeciesThermodynamicsPlan,
)
from phydrax.equations._conservation import (
    compile_conservation_problem,
    ConservationProblemIR,
)
from phydrax.equations._favre_les import (
    FavreLESFieldContract,
    FavreLESInputs,
    PreparedFavreLESModel,
)
from phydrax.equations._gas_dynamics import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
)
from phydrax.equations._homogeneous_thermodynamics import (
    HomogeneousHelmholtzPlan,
    IdealGasReferenceHelmholtzTerm,
    UNIVERSAL_GAS_CONSTANT,
    ZeroResidualHelmholtzTerm,
)
from phydrax.equations._les_closures import (
    AlgebraicLESInputs,
    LESFilterScale,
    LESParameterProvenance,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.equations.fem._nodal_conservation import (
    NodalDGConservationMethodPlan,
)
from phydrax.equations.fem._viscous_conservation import (
    ViscousBoundaryClosure,
    ViscousDGPlan,
)


def _schema():
    names = ("fuel", "oxidizer")
    return ChemicalSpeciesSchema.from_unique_species(
        names,
        (ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.020, 0.032)),
        ("F", "O"),
        jnp.eye(2, dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )


def _thermodynamics(schema=None):
    schema = _schema() if schema is None else schema
    calorics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.full((2, 1), 2.5 * UNIVERSAL_GAS_CONSTANT),
        jnp.asarray((1.0e3, 2.0e3)),
        reference_molar_entropy=jnp.asarray((100.0, 110.0)),
        reference_temperature=300.0,
        minimum_temperature=120.0,
        maximum_temperature=1500.0,
    )
    return HomogeneousHelmholtzPlan(
        IdealGasReferenceHelmholtzTerm(schema, calorics),
        ZeroResidualHelmholtzTerm(schema),
    )


def _fields(schema):
    return FavreLESFieldContract(schema.schema_id, schema.species_names)


def _closure(
    schema,
    coefficient=0.16,
    *,
    upper_bound=0.1,
    isotropic_trace_policy="neglected",
    filter_name="favre-cell-volume",
    dissipation_coefficient=1.05,
    kinetic_schmidt=1.0,
):
    resolved_filter = ResolvedLESFilter(
        filter_name,
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="open",
        scale_rule="volume-equivalent",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = LESParameterProvenance(
        resolved_filter,
        "three-dimensional-compressible-cell-transport",
        "variable-density-smooth-region",
        source_kind="user",
        evidence_ids=(),
    )
    algebraic = SmagorinskyLESPlan(coefficient).prepare(provenance)
    return PreparedFavreLESModel(
        algebraic,
        LESFilterScale(jnp.asarray((0.05, 0.06, 0.07))),
        _fields(schema),
        0.85,
        (("fuel", 0.7), ("oxidizer", 0.9)),
        upper_bound,
        isotropic_trace_policy=isotropic_trace_policy,
        sgs_kinetic_energy_dissipation_coefficient=dissipation_coefficient,
        sgs_kinetic_energy_turbulent_schmidt_number=kinetic_schmidt,
    )


def _inputs(
    fields,
    *,
    density=1.2,
    temperature=400.0,
    velocity=(1.0, -0.5, 0.2),
    velocity_gradient=((0.4, 0.1, 0.0), (-0.2, -0.1, 0.3), (0.0, 0.2, 0.05)),
    temperature_gradient=(2.0, -1.0, 0.5),
    mass_fractions=(0.3, 0.7),
    mass_fraction_gradient=((0.1, 0.02, 0.0), (-0.1, -0.02, 0.0)),
    specific_heat_capacity_pressure=1100.0,
    partial_specific_enthalpies=(2.0e5, 3.0e5),
    specific_sgs_kinetic_energy=None,
    specific_sgs_kinetic_energy_gradient=None,
):
    return FavreLESInputs(
        jnp.asarray(density),
        jnp.asarray(temperature),
        jnp.asarray(velocity),
        jnp.asarray(velocity_gradient),
        jnp.asarray(temperature_gradient),
        jnp.asarray(mass_fractions),
        jnp.asarray(mass_fraction_gradient),
        jnp.asarray(specific_heat_capacity_pressure),
        jnp.asarray(partial_specific_enthalpies),
        fields,
        specific_sgs_kinetic_energy=specific_sgs_kinetic_energy,
        specific_sgs_kinetic_energy_gradient=specific_sgs_kinetic_energy_gradient,
    )


def test_field_units_species_identity_and_filter_provenance_are_exact():
    schema = _schema()
    fields = _fields(schema)
    closure = _closure(schema)
    assert fields.density_unit == "kg/m^3"
    assert fields.temperature_unit == "K"
    assert fields.species_density_unit == "kg/m^3"
    assert fields.species_fraction_unit == "kg/kg"
    assert closure.provenance.resolved_filter.family == "implicit-grid-volume"
    assert closure.transport_role == "physical-subgrid-transport"
    assert not closure.numerical_stabilization_included
    assert closure.closure_id == _closure(schema).closure_id
    assert (
        closure.closure_id != _closure(schema, filter_name="other-cell-volume").closure_id
    )

    for changed_units in (
        {"density_unit": "g/cm^3"},
        {"temperature_unit": "degC"},
        {"species_density_unit": "mol/m^3"},
        {"species_fraction_unit": "mol/mol"},
    ):
        with pytest.raises(ValueError, match="exact canonical units"):
            FavreLESFieldContract(
                schema.schema_id,
                schema.species_names,
                **changed_units,
            )
    with pytest.raises(ValueError, match="exactly follow"):
        PreparedFavreLESModel(
            closure.algebraic_model,
            closure.filter_scale,
            fields,
            0.85,
            (("oxidizer", 0.9), ("fuel", 0.7)),
            0.1,
        )
    with pytest.raises(ValueError, match="three-dimensional"):
        closure.validate_compressible_transport_binding(
            schema.schema_id, schema.species_names, 2
        )


def test_constant_density_reduces_to_specific_model_and_is_objective():
    schema = _schema()
    closure = _closure(schema)
    inputs = _inputs(closure.fields)
    result = closure.evaluate(inputs)
    core = closure.algebraic_model.evaluate(
        AlgebraicLESInputs(inputs.favre_velocity_gradient, closure.filter_scale)
    )
    np.testing.assert_allclose(
        result.density_weighted_deviatoric_sgs_stress / inputs.density,
        core.specific_deviatoric_stress,
    )
    np.testing.assert_allclose(
        result.deviatoric_energy_transfer / inputs.density, core.energy_transfer
    )

    boost = jnp.asarray((4.0, -3.0, 2.0))
    boosted = closure.evaluate(
        _inputs(closure.fields, velocity=inputs.favre_velocity + boost)
    )
    np.testing.assert_allclose(boosted.sgs_stress, result.sgs_stress)
    np.testing.assert_allclose(boosted.sgs_heat_flux, result.sgs_heat_flux)
    np.testing.assert_allclose(boosted.sgs_species_flux, result.sgs_species_flux)
    np.testing.assert_allclose(
        boosted.stress_work_flux - result.stress_work_flux,
        -(boost @ result.sgs_stress),
    )

    rotation = jnp.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    rotated = closure.evaluate(
        _inputs(
            closure.fields,
            velocity=rotation @ inputs.favre_velocity,
            velocity_gradient=(rotation @ inputs.favre_velocity_gradient @ rotation.T),
            temperature_gradient=rotation @ inputs.temperature_gradient,
            mass_fraction_gradient=inputs.mass_fraction_gradient @ rotation.T,
        )
    )
    np.testing.assert_allclose(
        rotated.sgs_stress, rotation @ result.sgs_stress @ rotation.T
    )
    np.testing.assert_allclose(rotated.sgs_heat_flux, rotation @ result.sgs_heat_flux)
    np.testing.assert_allclose(
        rotated.sgs_species_flux, result.sgs_species_flux @ rotation.T
    )


def test_total_energy_stress_work_heat_and_named_species_flux_signs():
    schema = _schema()
    closure = _closure(schema)
    inputs = _inputs(closure.fields)
    result = closure.evaluate(inputs)

    np.testing.assert_allclose(
        result.stress_work_flux,
        -(inputs.favre_velocity @ result.sgs_stress),
    )
    np.testing.assert_allclose(
        result.sgs_enthalpy_flux,
        result.sgs_heat_flux + result.sgs_species_enthalpy_flux,
    )
    np.testing.assert_allclose(
        result.conservative_total_energy_flux,
        result.stress_work_flux - result.sgs_enthalpy_flux,
    )
    assert float(result.sgs_heat_flux[0]) < 0.0
    assert float(result.species_flux("fuel")[0]) < 0.0
    assert float(result.species_flux("oxidizer")[0]) > 0.0
    np.testing.assert_allclose(jnp.sum(result.sgs_species_flux, axis=-2), 0.0)
    np.testing.assert_allclose(result.conservative_species_flux, -result.sgs_species_flux)
    assert bool(jnp.all(result.input_evidence.successful))
    assert bool(jnp.all(result.evidence.successful))


def test_isotropic_trace_policy_is_explicit_and_conservatively_integrated():
    schema = _schema()
    provided = _closure(schema, isotropic_trace_policy="provided-sgs-kinetic-energy")
    assert provided.transport_role == "physical-subgrid-transport"
    assert not provided.numerical_stabilization_included
    kinetic_energy = jnp.asarray(0.4)
    kinetic_gradient = jnp.asarray((0.2, -0.1, 0.05))
    result = provided.evaluate(
        _inputs(
            provided.fields,
            specific_sgs_kinetic_energy=kinetic_energy,
            specific_sgs_kinetic_energy_gradient=kinetic_gradient,
        )
    )
    np.testing.assert_allclose(
        jnp.trace(result.isotropic_sgs_stress), 2.0 * 1.2 * kinetic_energy
    )
    np.testing.assert_allclose(
        result.conservative_total_energy_flux,
        result.conservative_resolved_energy_flux
        + result.sgs_kinetic_energy_diffusion_flux,
    )

    neglected = _closure(schema)
    np.testing.assert_array_equal(
        neglected.evaluate(_inputs(neglected.fields)).isotropic_sgs_stress,
        jnp.zeros((3, 3)),
    )
    with pytest.raises(ValueError, match="forbids SGS kinetic energy"):
        neglected.evaluate(
            _inputs(
                neglected.fields,
                specific_sgs_kinetic_energy=kinetic_energy,
                specific_sgs_kinetic_energy_gradient=kinetic_gradient,
            )
        )
    with pytest.raises(ValueError, match="requires SGS kinetic energy and its gradient"):
        provided.evaluate(
            _inputs(
                provided.fields,
                specific_sgs_kinetic_energy=kinetic_energy,
            )
        )

    system = HomogeneousMixtureCompressibleNavierStokesSystem(
        _thermodynamics(schema),
        ConstantTransport(0.0, 0.0),
        3,
        favre_les=provided,
    )
    baseline = HomogeneousMixtureCompressibleNavierStokesSystem(
        _thermodynamics(schema), ConstantTransport(0.0, 0.0), 3
    )
    primitive = jnp.asarray((0.36, 0.84, 1.0, -0.5, 0.2, 400.0, 0.4))
    state = system.primitive_to_conserved(primitive)
    base_state = baseline.primitive_to_conserved(primitive[:-1])
    assert system.component_names[-1] == "sgs_kinetic_energy"
    np.testing.assert_allclose(
        state[system.energy_index] - state[system.sgs_kinetic_energy_index],
        base_state[baseline.energy_index],
    )
    np.testing.assert_allclose(system.conserved_to_primitive(state), primitive)
    assert bool(system.admissible(state))

    coupled_flux = system.physical_flux(state, 0)
    base_flux = baseline.physical_flux(base_state, 0)
    isotropic_pressure = (2.0 / 3.0) * state[system.sgs_kinetic_energy_index]
    np.testing.assert_allclose(
        coupled_flux[system.momentum_slice] - base_flux[baseline.momentum_slice],
        jnp.asarray((isotropic_pressure, 0.0, 0.0)),
    )
    np.testing.assert_allclose(
        coupled_flux[system.sgs_kinetic_energy_index],
        state[system.sgs_kinetic_energy_index] * primitive[2],
    )
    assert float(system.max_wave_speed(state, state, 0)) > float(
        baseline.max_wave_speed(base_state, base_state, 0)
    )


def test_zero_coefficient_is_exact_and_invalid_or_unbounded_inputs_are_refused():
    schema = _schema()
    zero = _closure(schema, coefficient=0.0, upper_bound=0.0)
    result = zero.evaluate(_inputs(zero.fields))
    for value in (
        result.kinematic_eddy_viscosity,
        result.dynamic_eddy_viscosity,
        result.sgs_stress,
        result.sgs_heat_flux,
        result.sgs_species_flux,
        result.conservative_total_energy_flux,
    ):
        np.testing.assert_array_equal(value, jnp.zeros_like(value))

    closure = _closure(schema)
    for invalid in (
        _inputs(closure.fields, density=0.0),
        _inputs(closure.fields, temperature=-1.0),
        _inputs(closure.fields, mass_fractions=(0.2, 0.7)),
        _inputs(closure.fields, mass_fractions=(-0.1, 1.1)),
    ):
        with pytest.raises(
            (ValueError, eqx.EquinoxRuntimeError), match="finite positive density"
        ):
            output = closure.evaluate(invalid)
            jax.block_until_ready(output.conservative_total_energy_flux)

    bounded = _closure(schema, upper_bound=1.0e-12)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="exceeds its configured"
    ):
        output = bounded.evaluate(_inputs(bounded.fields))
        jax.block_until_ready(output.kinematic_eddy_viscosity)


def test_transported_sgs_energy_exchange_signs_and_positivity_restriction():
    schema = _schema()
    closure = _closure(
        schema,
        isotropic_trace_policy="provided-sgs-kinetic-energy",
        dissipation_coefficient=1.0,
        kinetic_schmidt=0.5,
    )
    kinetic_energy = jnp.asarray(0.25)
    zero_gradient = jnp.zeros((3,))
    shear = closure.evaluate(
        _inputs(
            closure.fields,
            velocity_gradient=((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            specific_sgs_kinetic_energy=kinetic_energy,
            specific_sgs_kinetic_energy_gradient=zero_gradient,
        )
    )
    assert float(shear.deviatoric_energy_transfer) > 0.0
    assert float(shear.sgs_kinetic_energy_dissipation) > 0.0
    assert bool(shear.evidence.dissipation_nonnegative)

    compression = closure.evaluate(
        _inputs(
            closure.fields,
            velocity_gradient=-0.2 * jnp.eye(3),
            specific_sgs_kinetic_energy=kinetic_energy,
            specific_sgs_kinetic_energy_gradient=zero_gradient,
        )
    )
    expansion = closure.evaluate(
        _inputs(
            closure.fields,
            velocity_gradient=0.2 * jnp.eye(3),
            specific_sgs_kinetic_energy=kinetic_energy,
            specific_sgs_kinetic_energy_gradient=zero_gradient,
        )
    )
    assert float(compression.isotropic_energy_transfer) > 0.0
    assert float(expansion.isotropic_energy_transfer) < 0.0

    pure_sink = closure.evaluate(
        _inputs(
            closure.fields,
            velocity_gradient=jnp.zeros((3, 3)),
            specific_sgs_kinetic_energy=kinetic_energy,
            specific_sgs_kinetic_energy_gradient=zero_gradient,
        )
    )
    timestep = pure_sink.source_positivity_timestep()
    np.testing.assert_allclose(
        pure_sink.sgs_kinetic_energy_density
        + timestep * pure_sink.sgs_kinetic_energy_source,
        0.0,
        atol=2.0e-7,
    )
    assert (
        float(
            pure_sink.sgs_kinetic_energy_density
            + 1.01 * timestep * pure_sink.sgs_kinetic_energy_source
        )
        < 0.0
    )
    assert (
        closure.maximum_kinematic_diffusivity()
        >= closure.kinematic_viscosity_upper_bound / 0.5
    )


def test_transported_zero_state_and_total_energy_exchange_ledger_are_exact():
    schema = _schema()
    zero_closure = _closure(
        schema,
        coefficient=0.0,
        upper_bound=0.0,
        isotropic_trace_policy="provided-sgs-kinetic-energy",
        dissipation_coefficient=0.0,
    )
    zero = zero_closure.evaluate(
        _inputs(
            zero_closure.fields,
            velocity_gradient=jnp.zeros((3, 3)),
            temperature_gradient=jnp.zeros((3,)),
            mass_fraction_gradient=jnp.zeros((2, 3)),
            specific_sgs_kinetic_energy=jnp.asarray(0.0),
            specific_sgs_kinetic_energy_gradient=jnp.zeros((3,)),
        )
    )
    for value in (
        zero.sgs_stress,
        zero.sgs_kinetic_energy_production,
        zero.sgs_kinetic_energy_dissipation,
        zero.sgs_kinetic_energy_source,
        zero.sgs_kinetic_energy_diffusion_flux,
        zero.conservative_total_energy_flux,
    ):
        np.testing.assert_array_equal(value, jnp.zeros_like(value))

    closure = _closure(
        schema,
        isotropic_trace_policy="provided-sgs-kinetic-energy",
    )
    system = HomogeneousMixtureCompressibleNavierStokesSystem(
        _thermodynamics(schema),
        ConstantTransport(0.0, 0.0),
        3,
        favre_les=closure,
    )

    def state_at(point):
        density = 1.2 + 0.02 * point[0]
        fuel = 0.3 + 0.01 * point[1]
        species_density = density * jnp.stack((fuel, 1.0 - fuel))
        velocity = jnp.asarray(
            (
                1.0 + 0.4 * point[0] + 0.1 * point[1],
                -0.5 - 0.2 * point[0] - 0.1 * point[1] + 0.3 * point[2],
                0.2 + 0.2 * point[1] + 0.05 * point[2],
            )
        )
        temperature = 400.0 + 2.0 * point[0] - point[1] + 0.5 * point[2]
        kinetic = 0.25 + 0.02 * point[0] - 0.01 * point[2]
        return system.primitive_to_conserved(
            jnp.concatenate((species_density, velocity, temperature[None], kinetic[None]))
        )

    point = jnp.zeros((3,))
    state = state_at(point)
    gradient = jax.jacfwd(state_at)(point)
    flux, rate = system.viscous_flux_and_favre_rate(state, gradient)
    assert rate is not None
    transport = rate.transport
    np.testing.assert_allclose(
        flux[system.energy_index] - flux[system.sgs_kinetic_energy_index],
        transport.deviatoric_stress_work_flux - transport.sgs_enthalpy_flux,
    )
    np.testing.assert_allclose(
        flux[system.sgs_kinetic_energy_index],
        transport.sgs_kinetic_energy_diffusion_flux,
    )
    np.testing.assert_array_equal(
        rate.conserved_source[system.energy_index], jnp.asarray(0.0)
    )
    np.testing.assert_allclose(
        rate.conserved_source[system.sgs_kinetic_energy_index],
        transport.sgs_kinetic_energy_source,
    )
    np.testing.assert_array_equal(rate.total_energy_source, jnp.asarray(0.0))
    non_sgs = jnp.delete(rate.conserved_source, system.sgs_kinetic_energy_index)
    np.testing.assert_array_equal(non_sgs, jnp.zeros_like(non_sgs))

    invalid = state.at[system.sgs_kinetic_energy_index].set(-1.0e-3)
    assert not bool(system.admissible(invalid))
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="nonnegative species and SGS-energy"
    ):
        output = system.viscous_flux(invalid, gradient)
        jax.block_until_ready(output)


def test_jit_and_jvp_preserve_favre_transport_contract():
    schema = _schema()
    closure = _closure(schema)
    base_gradient = jnp.asarray(((0.4, 0.1, 0.0), (-0.2, -0.1, 0.3), (0.0, 0.2, 0.05)))

    def energy_flux(velocity_gradient):
        return closure.evaluate(
            _inputs(closure.fields, velocity_gradient=velocity_gradient)
        ).conservative_total_energy_flux

    eager = energy_flux(base_gradient)
    compiled = jax.jit(energy_flux)(base_gradient)
    np.testing.assert_allclose(compiled, eager)
    primal, tangent = jax.jvp(
        energy_flux,
        (base_gradient,),
        (jnp.full_like(base_gradient, 0.01),),
    )
    np.testing.assert_allclose(primal, eager)
    assert bool(jnp.all(jnp.isfinite(tangent)))
    assert float(jnp.sqrt(jnp.sum(tangent * tangent))) > 0.0


def test_transported_coupled_state_is_restartable_jittable_and_differentiable():
    schema = _schema()
    system = HomogeneousMixtureCompressibleNavierStokesSystem(
        _thermodynamics(schema),
        ConstantTransport(0.0, 0.0),
        3,
        favre_les=_closure(
            schema,
            isotropic_trace_policy="provided-sgs-kinetic-energy",
        ),
    )
    primitive = jnp.asarray((0.36, 0.84, 1.0, -0.5, 0.2, 400.0, 0.25))
    state = system.primitive_to_conserved(primitive)
    restarted = jnp.asarray(np.asarray(state).copy())
    np.testing.assert_array_equal(restarted, state)
    np.testing.assert_allclose(
        system.conserved_to_primitive(restarted), primitive, rtol=2.0e-6
    )

    def coupled_terms(local_state):
        gradient = jnp.zeros(local_state.shape + (3,), dtype=local_state.dtype)
        viscous, rate = system.viscous_flux_and_favre_rate(local_state, gradient)
        return jnp.concatenate(
            (
                system.physical_flux(local_state, 0),
                viscous.reshape((-1,)),
                rate.conserved_source,
            )
        )

    eager = coupled_terms(state)
    compiled = jax.jit(coupled_terms)(state)
    np.testing.assert_allclose(compiled, eager, rtol=2.0e-6)
    primal, tangent = jax.jvp(
        coupled_terms,
        (state,),
        (jnp.linspace(1.0e-4, 7.0e-4, system.component_count),),
    )
    np.testing.assert_allclose(primal, eager, rtol=2.0e-6)
    assert bool(jnp.all(jnp.isfinite(tangent)))
    assert float(jnp.linalg.norm(tangent)) > 0.0

    left, right, speeds = system.eigensystem(state, state, 0)
    np.testing.assert_allclose(
        left @ right,
        jnp.eye(system.component_count),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    expected_sound = jnp.sqrt(
        system.recover_thermodynamics(state).state.frozen_sound_speed_squared
        + (2.0 / 3.0) * primitive[-1]
    )
    np.testing.assert_allclose(speeds[-1] - primitive[2], expected_sound, rtol=2.0e-6)


def test_gas_viscous_flux_adds_favre_transport_without_touching_stabilization():
    schema = _schema()
    thermodynamics = _thermodynamics(schema)
    closure = _closure(schema)
    baseline = HomogeneousMixtureCompressibleNavierStokesSystem(
        thermodynamics, ConstantTransport(0.0, 0.0), 3
    )
    system = HomogeneousMixtureCompressibleNavierStokesSystem(
        thermodynamics,
        ConstantTransport(0.0, 0.0),
        3,
        favre_les=closure,
    )

    def state_at(point):
        density = 1.2 + 0.02 * point[0]
        fuel = 0.3 + 0.01 * point[1]
        species_density = density * jnp.stack((fuel, 1.0 - fuel))
        velocity = jnp.asarray((1.0, -0.5, 0.2)) + jnp.asarray(
            (
                0.4 * point[0] + 0.1 * point[1],
                -0.2 * point[0] - 0.1 * point[1] + 0.3 * point[2],
                0.2 * point[1] + 0.05 * point[2],
            )
        )
        temperature = 400.0 + 2.0 * point[0] - point[1] + 0.5 * point[2]
        return system.primitive_to_conserved(
            jnp.concatenate((species_density, velocity, temperature[None]))
        )

    point = jnp.asarray((0.0, 0.0, 0.0))
    state = state_at(point)
    gradient = jax.jacfwd(state_at)(point)
    favre = system.favre_les_transport(state, gradient)
    flux = system.viscous_flux(state, gradient)
    np.testing.assert_allclose(flux[:2], favre.conservative_species_flux)
    np.testing.assert_allclose(flux[2:5], favre.conservative_momentum_flux)
    np.testing.assert_allclose(flux[5], favre.conservative_total_energy_flux)
    assert (
        float(system.maximum_diffusivity(state))
        >= closure.maximum_kinematic_diffusivity()
    )

    for axis in range(3):
        np.testing.assert_allclose(
            system.physical_flux(state, axis), baseline.physical_flux(state, axis)
        )
    np.testing.assert_allclose(
        system.max_wave_speed(state, state, 0),
        baseline.max_wave_speed(state, state, 0),
    )
    assert system.system_id != baseline.system_id
    with pytest.raises(ValueError, match="primitive-gradient-only"):
        system.viscous_flux_from_primitive_gradients(
            jnp.ones((3,)),
            jnp.eye(3),
            jnp.ones((3,)),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        )


def test_existing_manufactured_compressible_semidiscrete_path_carries_favre_flux():
    schema = _schema()
    system = HomogeneousMixtureCompressibleNavierStokesSystem(
        _thermodynamics(schema),
        ConstantTransport(0.01, 0.02),
        3,
        favre_les=_closure(schema),
    )

    def exact_state(time, point, args):
        del args
        phase = point[0] + 0.5 * point[1] - 0.25 * point[2] - time
        density = 1.2 + 0.01 * jnp.sin(phase)
        fuel = 0.3 + 0.005 * jnp.cos(phase)
        species_density = density * jnp.stack((fuel, 1.0 - fuel))
        velocity = jnp.asarray(
            (
                0.8 + 0.03 * jnp.cos(phase),
                -0.2 + 0.02 * jnp.sin(phase),
                0.1 - 0.01 * jnp.cos(phase),
            )
        )
        temperature = 400.0 + 2.0 * jnp.sin(phase)
        return system.primitive_to_conserved(
            jnp.concatenate((species_density, velocity, temperature[None]))
        )

    evidence = ManufacturedViscousNSPlan(
        3, exact_state, "favre-les-smooth-manufactured-state"
    ).evaluate(system, 0.2, jnp.asarray(((0.1, 0.2, 0.3),)))
    assert bool(evidence.finite)
    np.testing.assert_allclose(evidence.identity_residual, 0.0, atol=2.0e-6)
    assert bool(jnp.all(system.admissible(evidence.state)))


def test_nodal_dg_semidiscrete_rhs_includes_gradient_aware_sgs_energy_rate():
    schema = _schema()
    system = HomogeneousMixtureCompressibleNavierStokesSystem(
        _thermodynamics(schema),
        ConstantTransport(0.0, 0.0),
        3,
        favre_les=_closure(
            schema,
            coefficient=0.0,
            upper_bound=0.0,
            isotropic_trace_policy="provided-sgs-kinetic-energy",
            dissipation_coefficient=1.0,
        ),
    )
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "cells",
                "tetrahedron",
                np.arange(4, dtype=np.int32)[None, :],
            ),
        ),
    )
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state",
            discontinuous_element("tetrahedron", 1),
            component_shape=(system.component_count,),
        ),
    ).prepare()
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = FiniteElementBoundarySet(
        discretization,
        {"outflow": (exterior, ExtrapolationBoundary())},
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR(
            "transported-favre-sgs-energy",
            "state",
            system,
            boundaries,
        ),
        discretization,
        NodalDGConservationMethodPlan(
            RusanovFluxPlan(),
            viscous=ViscousDGPlan(
                formulation="ldg",
                boundary_closures=(
                    ViscousBoundaryClosure(boundaries.patches[0].boundary.boundary_id),
                ),
            ),
        ),
    )
    point_state = system.primitive_to_conserved(
        jnp.asarray((0.36, 0.84, 0.0, 0.0, 0.0, 400.0, 0.25))
    )
    state = jnp.broadcast_to(
        point_state, discretization.field_spaces[0].vector_space.shape
    )
    rate = compiled(0.0, state)
    local_expected = system.favre_les_coupled_rate(
        point_state,
        jnp.zeros(point_state.shape + (3,)),
    ).conserved_source
    np.testing.assert_allclose(
        rate,
        jnp.broadcast_to(local_expected, rate.shape),
        atol=2.0e-4,
        rtol=2.0e-5,
    )
    np.testing.assert_allclose(rate[..., system.energy_index], 0.0, atol=2.0e-4)
    assert bool(jnp.all(rate[..., system.sgs_kinetic_energy_index] < 0.0))

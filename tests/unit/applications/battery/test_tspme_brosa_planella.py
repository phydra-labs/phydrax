#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.battery._experiment import (
    BatteryDiffraxSolvePlan,
    BatteryExperimentPlan,
    BatteryOutputPlan,
    BatteryRuntimeInputs,
)
from phydrax.applications.battery._properties import (
    ConcentrationTemperaturePropertyLaw,
    ConstantPropertyLaw,
    TabulatedPropertyLaw,
)
from phydrax.applications.battery._protocol import (
    BatteryProtocolPlan,
    BatteryProtocolValues,
    CurrentStepPlan,
    RestStepPlan,
)
from phydrax.applications.battery._results import BatteryRunStatus
from phydrax.applications.battery._spm import SpmParameters
from phydrax.applications.battery._spme_marquis2019 import (
    Marquis2019SpmeInitialCondition,
    Marquis2019SpmeParameters,
    Marquis2019SpmePlan,
)
from phydrax.applications.battery._tspme_brosa_planella import (
    BrosaPlanellaTspmeAdapter,
    BrosaPlanellaTspmeInitialCondition,
    BrosaPlanellaTspmeParameters,
    BrosaPlanellaTspmePlan,
    BrosaPlanellaTspmeState,
)
from phydrax.qualification import CapabilityProfile, SupportTuple


def _constant(
    value, support, *, quantity, coordinate, value_unit, coordinate_unit, positive=True
):
    return ConstantPropertyLaw(
        jnp.asarray(value),
        jnp.asarray(support),
        value_bounds=(0.0, jnp.inf) if positive else (-jnp.inf, jnp.inf),
        quantity=quantity,
        coordinate=coordinate,
        value_unit=value_unit,
        coordinate_unit=coordinate_unit,
        source_id=f"test:tspme:{quantity}:constant",
    )


def _table(
    nodes, values, *, quantity, coordinate, value_unit, coordinate_unit, positive=True
):
    return TabulatedPropertyLaw(
        jnp.asarray(nodes),
        jnp.asarray(values),
        value_bounds=(0.0, jnp.inf) if positive else (-jnp.inf, jnp.inf),
        quantity=quantity,
        coordinate=coordinate,
        value_unit=value_unit,
        coordinate_unit=coordinate_unit,
        source_id=f"test:tspme:{quantity}:table",
    )


def _temperature_table(values, *, quantity, value_unit):
    return _table(
        (280.0, 300.0, 320.0),
        values,
        quantity=quantity,
        coordinate="temperature",
        value_unit=value_unit,
        coordinate_unit="K",
    )


def _bivariate(values, *, quantity, value_unit, value_bounds=(0.0, jnp.inf)):
    return ConcentrationTemperaturePropertyLaw(
        jnp.asarray((500.0, 1000.0, 1500.0)),
        jnp.asarray((280.0, 300.0, 320.0)),
        jnp.asarray(values),
        value_bounds=value_bounds,
        quantity=quantity,
        value_unit=value_unit,
        source_id=f"test:tspme:{quantity}:bivariate",
    )


def _spm_parameters(*, maximum_current=1.0):
    return SpmParameters(
        electrode_area_m2=0.1,
        negative_electrode_thickness_m=1.0e-4,
        positive_electrode_thickness_m=1.0e-4,
        negative_active_material_volume_fraction=0.6,
        positive_active_material_volume_fraction=0.6,
        negative_particle_radius_m=5.0e-6,
        positive_particle_radius_m=4.0e-6,
        negative_maximum_concentration_mol_m3=3.0e4,
        positive_maximum_concentration_mol_m3=3.0e4,
        temperature_k=300.0,
        maximum_absolute_current_a=maximum_current,
        negative_stoichiometry_at_empty=0.1,
        negative_stoichiometry_at_full=0.9,
        positive_stoichiometry_at_empty=0.9,
        positive_stoichiometry_at_full=0.1,
        negative_solid_diffusivity=_temperature_table(
            (1.0e-14, 2.0e-14, 4.0e-14),
            quantity="negative-solid-diffusivity",
            value_unit="m2/s",
        ),
        positive_solid_diffusivity=_temperature_table(
            (0.8e-14, 1.5e-14, 3.0e-14),
            quantity="positive-solid-diffusivity",
            value_unit="m2/s",
        ),
        negative_exchange_current_density=_temperature_table(
            (3.0, 5.0, 8.0), quantity="negative-exchange-current", value_unit="A/m2"
        ),
        positive_exchange_current_density=_temperature_table(
            (2.0, 4.0, 7.0), quantity="positive-exchange-current", value_unit="A/m2"
        ),
        negative_open_circuit_potential=_constant(
            0.1,
            (0.0, 1.0),
            quantity="negative-ocp-reference",
            coordinate="stoichiometry",
            value_unit="V",
            coordinate_unit="1",
        ),
        positive_open_circuit_potential=_constant(
            4.1,
            (0.0, 1.0),
            quantity="positive-ocp-reference",
            coordinate="stoichiometry",
            value_unit="V",
            coordinate_unit="1",
        ),
    )


def _marquis_parameters(*, maximum_current=1.0):
    return Marquis2019SpmeParameters(
        _spm_parameters(maximum_current=maximum_current),
        separator_thickness_m=5.0e-5,
        negative_electrolyte_porosity=0.3,
        separator_electrolyte_porosity=0.5,
        positive_electrolyte_porosity=0.3,
        bruggeman_coefficient=1.5,
        typical_electrolyte_concentration_mol_m3=1000.0,
        electrolyte_diffusivity=_bivariate(
            jnp.full((3, 3), 2.0e-10),
            quantity="marquis-electrolyte-diffusivity",
            value_unit="m2/s",
        ),
        electrolyte_conductivity=_bivariate(
            jnp.full((3, 3), 1.2),
            quantity="marquis-electrolyte-conductivity",
            value_unit="S/m",
        ),
        transference_number=_bivariate(
            jnp.full((3, 3), 0.4),
            quantity="marquis-transference-number",
            value_unit="1",
            value_bounds=(0.0, 1.0),
        ),
        negative_solid_conductivity_s_m=100.0,
        positive_solid_conductivity_s_m=80.0,
    )


def _parameters(*, maximum_current=1.0):
    return BrosaPlanellaTspmeParameters(
        _marquis_parameters(maximum_current=maximum_current),
        ambient_temperature_k=300.0,
        volumetric_heat_capacity_j_m3_k=2.0e6,
        heat_transfer_coefficient_w_m2_k=1.0,
        cooling_surface_area_per_volume_m_inv=100.0,
        battery_length_scale_m=1.0e-3,
        battery_thermal_conductivity_w_m_k=1.0,
        typical_discharge_time_s=1000.0,
        typical_electrode_potential_v=4.0,
        negative_solid_conductivity=_temperature_table(
            (120.0, 100.0, 80.0), quantity="negative-solid-conductivity", value_unit="S/m"
        ),
        positive_solid_conductivity=_temperature_table(
            (100.0, 80.0, 60.0), quantity="positive-solid-conductivity", value_unit="S/m"
        ),
        electrolyte_diffusivity=_bivariate(
            (
                (0.75e-10, 1.5e-10, 3.0e-10),
                (1.0e-10, 2.0e-10, 4.0e-10),
                (1.25e-10, 2.5e-10, 5.0e-10),
            ),
            quantity="electrolyte-diffusivity",
            value_unit="m2/s",
        ),
        electrolyte_conductivity=_bivariate(
            (
                (0.56, 0.70, 0.91),
                (0.96, 1.20, 1.56),
                (1.20, 1.50, 1.95),
            ),
            quantity="electrolyte-conductivity",
            value_unit="S/m",
        ),
        electrolyte_transference_number=_bivariate(
            (
                (0.33, 0.35, 0.37),
                (0.38, 0.40, 0.42),
                (0.43, 0.45, 0.47),
            ),
            quantity="electrolyte-transference-number",
            value_unit="1",
            value_bounds=(0.0, 1.0),
        ),
        electrolyte_thermodynamic_factor=_table(
            (500.0, 1000.0, 1500.0),
            (1.0, 1.5, 2.0),
            quantity="electrolyte-thermodynamic-factor",
            coordinate="electrolyte_concentration",
            value_unit="1",
            coordinate_unit="mol/m3",
        ),
        negative_entropic_coefficient=_constant(
            -0.5e-4,
            (0.0, 1.0),
            quantity="negative-entropic-coefficient",
            coordinate="stoichiometry",
            value_unit="V/K",
            coordinate_unit="1",
            positive=False,
        ),
        positive_entropic_coefficient=_constant(
            1.0e-4,
            (0.0, 1.0),
            quantity="positive-entropic-coefficient",
            coordinate="stoichiometry",
            value_unit="V/K",
            coordinate_unit="1",
            positive=False,
        ),
    )


def _adapter(*, small_threshold=0.1):
    marquis_plan = Marquis2019SpmePlan(
        5,
        negative_electrolyte_cell_count=4,
        separator_electrolyte_cell_count=3,
        positive_electrolyte_cell_count=4,
    )
    adapter = BrosaPlanellaTspmeAdapter(
        BrosaPlanellaTspmePlan(
            marquis_plan, applicability_small_parameter_threshold=small_threshold
        )
    )
    return adapter, adapter.prepare(), _parameters()


def _state(adapter, prepared, parameters):
    return adapter.initial_state(
        prepared,
        parameters,
        BrosaPlanellaTspmeInitialCondition(Marquis2019SpmeInitialCondition(0.5, 0.5)),
    )


def _runtime(parameters, current, *, duration=1.0):
    protocol = BatteryProtocolPlan((CurrentStepPlan(duration),))
    values = BatteryProtocolValues(protocol, jnp.asarray((current,)))
    return BatteryRuntimeInputs(
        parameters,
        protocol.input_policy(values, node_side="right"),
        values.stop_thresholds,
        protocol_id=protocol.protocol_id,
        observation_input_policy=protocol.input_policy(values),
    )


def _nonuniform_electrolyte_state(prepared, state):
    mesh = prepared.spme.through_cell
    concentration = jnp.where(
        mesh.negative_mask, 900.0, jnp.where(mesh.positive_mask, 1100.0, 1000.0)
    )
    return BrosaPlanellaTspmeState(
        eqx.tree_at(
            lambda value: value.electrolyte_amount_mol,
            state.spme_state,
            state.spme_state.electrolyte_amount_mol * concentration / 1000.0,
        ),
        state.temperature_k,
    )


def test_exact_section_three_identity_zero_current_and_isothermal_spme_reduction():
    adapter, prepared, parameters = _adapter()
    state = _state(adapter, prepared, parameters)
    evaluation = prepared.evaluate(state, parameters, 0.0)
    assert adapter.model_id == "battery:tspme:brosa-planella:base-prescribed-current"
    assert adapter.source_formulation_id == "arxiv:2011.01611v3:section-3:eqs-1-10"
    assert adapter.plan.source_formulation_id == adapter.source_formulation_id
    assert prepared.spme.plan.plan_id == adapter.plan.spme_plan.plan_id
    assert bool(evaluation.domain_valid)
    np.testing.assert_allclose(evaluation.temperature_rate_k_s, 0.0, atol=0.0)
    np.testing.assert_allclose(evaluation.boundary_cooling_w_m3, 0.0, atol=0.0)
    np.testing.assert_allclose(evaluation.generated_heat_w_m3, 0.0, atol=0.0)
    np.testing.assert_allclose(evaluation.particle_ocp_temperature_correction_v, 0.0)
    np.testing.assert_allclose(evaluation.voltage_v, 4.0)

    thermal = prepared.evaluate(state, parameters, 0.2)
    isothermal = prepared.spme.evaluate(state.spme_state, parameters.spme_parameters, 0.2)
    np.testing.assert_allclose(
        thermal.particle_transport.negative.amount_rate_mol_s,
        isothermal.particle_transport.negative.amount_rate_mol_s,
    )
    np.testing.assert_allclose(
        thermal.electrolyte_transport.amount_rate_mol_s,
        isothermal.electrolyte_transport.amount_rate_mol_s,
    )
    np.testing.assert_allclose(thermal.voltage_v, isothermal.voltage_v, rtol=2.0e-6)
    assert (
        parameters.electrochemical_property_support_id
        != parameters.thermal_property_support_id
    )
    assert adapter.plan.applicability_support_id != adapter.plan.spme_plan.plan_id


def test_temperature_is_two_way_coupled_to_every_property_family_and_ocp():
    adapter, prepared, parameters = _adapter()
    initial = _state(adapter, prepared, parameters)
    cold = prepared.evaluate(
        BrosaPlanellaTspmeState(initial.spme_state, 290.0), parameters, 0.2
    )
    hot = prepared.evaluate(
        BrosaPlanellaTspmeState(initial.spme_state, 310.0), parameters, 0.2
    )
    assert float(hot.negative_solid_diffusivity_m2_s) > float(
        cold.negative_solid_diffusivity_m2_s
    )
    assert float(hot.positive_solid_diffusivity_m2_s) > float(
        cold.positive_solid_diffusivity_m2_s
    )
    assert float(hot.electrolyte_diffusivity_m2_s) > float(
        cold.electrolyte_diffusivity_m2_s
    )
    assert float(hot.negative_solid_conductivity_s_m) < float(
        cold.negative_solid_conductivity_s_m
    )
    assert float(hot.positive_solid_conductivity_s_m) < float(
        cold.positive_solid_conductivity_s_m
    )
    assert float(hot.typical_electrolyte_conductivity_s_m) > float(
        cold.typical_electrolyte_conductivity_s_m
    )
    assert float(hot.negative_exchange_current_density_a_m2) > float(
        cold.negative_exchange_current_density_a_m2
    )
    assert float(hot.positive_exchange_current_density_a_m2) > float(
        cold.positive_exchange_current_density_a_m2
    )
    assert float(hot.particle_ocp_temperature_correction_v) > float(
        cold.particle_ocp_temperature_correction_v
    )
    assert not np.isclose(float(hot.voltage_v), float(cold.voltage_v))
    assert float(hot.boundary_cooling_w_m3) > 0.0
    assert float(cold.boundary_cooling_w_m3) < 0.0
    assert float(prepared.evaluate(initial, parameters, 0.2).temperature_rate_k_s) != 0.0


def test_every_section_three_heat_term_sign_and_power_identity_is_explicit():
    adapter, prepared, parameters = _adapter()
    state = _nonuniform_electrolyte_state(prepared, _state(adapter, prepared, parameters))
    state = BrosaPlanellaTspmeState(state.spme_state, 305.0)
    charge = prepared.evaluate(state, parameters, 0.2)
    discharge = prepared.evaluate(state, parameters, -0.2)
    expected_primitive_difference = (
        0.3475 * np.log(1100.0 / 900.0)
        + 0.000645 * (1100.0 - 900.0)
        - 0.5e-7 * (1100.0**2 - 900.0**2)
    )
    expected_concentration_overpotential = (
        2.0 * 8.31446261815324 * 305.0 / 96485.33212 * expected_primitive_difference
    )
    np.testing.assert_allclose(
        charge.electrolyte_concentration_overpotential_v,
        expected_concentration_overpotential,
    )
    ideal_parameters = eqx.tree_at(
        lambda value: value.electrolyte_thermodynamic_factor.values,
        parameters,
        jnp.ones((3,)),
    )
    ideal = prepared.evaluate(state, ideal_parameters, 0.2)
    assert not np.isclose(
        float(charge.electrolyte_concentration_overpotential_v),
        float(ideal.electrolyte_concentration_overpotential_v),
    )
    constant_diffusivity_parameters = eqx.tree_at(
        lambda value: value.electrolyte_diffusivity.values,
        parameters,
        jnp.full((3, 3), 2.0e-10),
    )
    constant_diffusivity = prepared.evaluate(state, constant_diffusivity_parameters, 0.2)
    assert (
        float(
            jnp.max(
                jnp.abs(
                    charge.electrolyte_transport.amount_rate_mol_s
                    - constant_diffusivity.electrolyte_transport.amount_rate_mol_s
                )
            )
        )
        > 0.0
    )
    assert bool(charge.electrolyte_transport_support_valid)
    assert bool(charge.thermodynamic_factor_support_valid)
    assert (
        parameters.electrolyte_thermodynamic_factor.law_id
        != parameters.electrolyte_conductivity.law_id
    )
    unsupported_state = BrosaPlanellaTspmeState(
        eqx.tree_at(
            lambda value: value.electrolyte_amount_mol,
            state.spme_state,
            state.spme_state.electrolyte_amount_mol * 0.4,
        ),
        state.temperature_k,
    )
    unsupported = prepared.evaluate(unsupported_state, parameters, 0.2)
    assert not bool(unsupported.thermodynamic_factor_support_valid)
    assert not bool(unsupported.property_support_valid)
    assert not bool(unsupported.domain_valid)
    expected_solid = (
        2.0**2
        / (3.0 * 2.5e-4)
        * (
            1.0e-4 / charge.negative_solid_conductivity_s_m
            + 1.0e-4 / charge.positive_solid_conductivity_s_m
        )
    )
    np.testing.assert_allclose(charge.solid_ohmic_heat_w_m3, expected_solid)
    assert float(charge.solid_ohmic_heat_w_m3) > 0.0
    assert float(discharge.solid_ohmic_heat_w_m3) > 0.0
    assert float(charge.electrolyte_concentration_heat_w_m3) > 0.0
    assert float(discharge.electrolyte_concentration_heat_w_m3) < 0.0
    assert float(charge.electrolyte_ohmic_heat_w_m3) > 0.0
    assert float(discharge.electrolyte_ohmic_heat_w_m3) > 0.0
    assert float(charge.irreversible_reaction_heat_w_m3) > 0.0
    assert float(discharge.irreversible_reaction_heat_w_m3) > 0.0
    assert float(charge.reversible_reaction_heat_w_m3) > 0.0
    assert float(discharge.reversible_reaction_heat_w_m3) < 0.0
    assert float(charge.boundary_cooling_w_m3) > 0.0
    np.testing.assert_allclose(
        charge.net_heat_w_m3, charge.generated_heat_w_m3 - charge.boundary_cooling_w_m3
    )
    np.testing.assert_allclose(
        charge.generated_heat_w_m3,
        charge.solid_ohmic_heat_w_m3
        + charge.electrolyte_concentration_heat_w_m3
        + charge.electrolyte_ohmic_heat_w_m3
        + charge.irreversible_reaction_heat_w_m3
        + charge.reversible_reaction_heat_w_m3,
    )
    np.testing.assert_allclose(
        charge.voltage_v,
        charge.particle_ocp_v
        + charge.reaction_overpotential_v
        + charge.electrolyte_concentration_overpotential_v
        + charge.electrolyte_ohmic_loss_v
        + charge.solid_ohmic_loss_v,
    )
    np.testing.assert_allclose(charge.heat_sum_residual_w_m3, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        charge.solid_power_identity_residual_w_m3, 0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        charge.electrolyte_power_identity_residual_w_m3, 0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        charge.reaction_power_identity_residual_w_m3, 0.0, atol=1.0e-12
    )


def test_section_three_applicability_is_independent_and_refuses_bad_regimes():
    adapter, prepared, parameters = _adapter()
    state = _state(adapter, prepared, parameters)
    accepted = prepared.evaluate(state, parameters, 0.1)
    assert float(accepted.small_overpotential_ratio) < 0.1
    assert float(accepted.negative_solid_conductivity_number) >= 1.0
    assert float(accepted.positive_solid_conductivity_number) >= 1.0
    assert float(accepted.electrolyte_conductivity_number) >= 1.0
    assert float(accepted.biot_number) < 0.1
    assert float(accepted.internal_conduction_to_cooling_ratio) > 10.0
    assert float(accepted.internal_thermal_conduction_number) > 10.0
    assert bool(accepted.applicability_conditions_satisfied)
    bad_electrochemical = eqx.tree_at(
        lambda value: value.typical_electrode_potential_v, parameters, jnp.asarray(0.01)
    )
    refused_electrochemical = prepared.evaluate(state, bad_electrochemical, 0.1)
    assert not bool(refused_electrochemical.electrochemical_applicability_satisfied)
    assert not bool(refused_electrochemical.applicability_conditions_satisfied)
    assert bool(refused_electrochemical.domain_valid)
    bad_thermal = eqx.tree_at(
        lambda value: value.battery_thermal_conductivity_w_m_k,
        parameters,
        jnp.asarray(1.0e-4),
    )
    refused_thermal = prepared.evaluate(state, bad_thermal, 0.1)
    assert not bool(refused_thermal.thermal_applicability_satisfied)
    assert not bool(refused_thermal.applicability_conditions_satisfied)
    assert bool(refused_thermal.domain_valid)


def test_problem_observation_jit_vmap_and_gradients_cover_both_coupling_directions():
    adapter, prepared, parameters = _adapter()
    state = _state(adapter, prepared, parameters)
    runtime = _runtime(parameters, 0.2)
    problem = adapter.problem(prepared, state, runtime)
    eager = problem.drift(jnp.asarray(0.0), state, runtime)
    compiled = jax.jit(lambda time, candidate: problem.drift(time, candidate, runtime))(
        jnp.asarray(0.0), state
    )
    np.testing.assert_allclose(
        compiled.spme_state.negative_amount_mol,
        eager.spme_state.negative_amount_mol,
        atol=2.0e-19,
    )
    np.testing.assert_allclose(compiled.temperature_k, eager.temperature_k)
    compiled_output = jax.jit(
        lambda candidate: adapter.observe(prepared, jnp.asarray(0.0), candidate, runtime)
    )(state)
    assert bool(compiled_output.domain_valid)
    states = jax.tree.map(lambda value: jnp.stack((value, value)), state)
    mapped = adapter.observe(prepared, jnp.asarray((0.0, 0.0)), states, runtime)
    assert mapped.values.shape == (2, len(adapter.observable_names))
    assert bool(jnp.all(mapped.domain_valid))

    def voltage_for_temperature(temperature):
        return prepared.evaluate(
            BrosaPlanellaTspmeState(state.spme_state, temperature), parameters, 0.2
        ).voltage_v

    def temperature_rate_for_current(current):
        return prepared.evaluate(state, parameters, current).temperature_rate_k_s

    temperature_gradient = jax.grad(voltage_for_temperature)(jnp.asarray(300.0))
    current_gradient = jax.grad(temperature_rate_for_current)(jnp.asarray(0.2))
    assert bool(jnp.isfinite(temperature_gradient))
    assert bool(jnp.isfinite(current_gradient))
    assert float(temperature_gradient) != 0.0
    assert float(current_gradient) != 0.0


def test_protocol_orchestration_returns_separate_thermal_and_electrochemical_ledgers():
    parameters = _parameters()
    adapter = BrosaPlanellaTspmeAdapter(
        BrosaPlanellaTspmePlan(
            Marquis2019SpmePlan(
                5,
                negative_electrolyte_cell_count=3,
                separator_electrolyte_cell_count=2,
                positive_electrolyte_cell_count=3,
            ),
            ledger_energy_absolute_tolerance_j=2.0e-4,
        )
    )
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(0.25), RestStepPlan(0.25), CurrentStepPlan(0.25))
    )
    protocol_values = BatteryProtocolValues(protocol, jnp.asarray((0.02, -0.01)))
    support = SupportTuple(
        "battery.simulation",
        {
            "model": "tspme-brosa-planella-base",
            "control": "prescribed-current",
            "thermal": "homogeneous",
        },
    )
    profile = CapabilityProfile(
        "battery.tspme-brosa-planella-test-candidate",
        "phydrax-tests",
        "candidate",
        (support,),
        released=False,
    )
    save_times = jnp.asarray((0.0, 0.125, 0.25, 0.5, 0.625, 0.75))
    experiment = BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(
            (
                "voltage_v",
                "temperature_k",
                "heat:generated_w_m3",
                "applicability:conditions_satisfied",
                "current_a",
            )
        ),
        BatteryDiffraxSolvePlan(
            stepsize_controller=dfx.StepTo(ts=save_times),
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
        ),
        save_times,
        profile,
        support,
    ).prepare()
    result = experiment.run(
        parameters,
        BrosaPlanellaTspmeInitialCondition(Marquis2019SpmeInitialCondition(0.5, 0.5)),
        protocol_values,
    )
    solution = result.native_solution
    interval_duration = solution.times[1:] - solution.times[:-1]
    interval_midpoints = 0.5 * (solution.times[:-1] + solution.times[1:])
    interval_currents = jax.vmap(lambda time: protocol.current(time, protocol_values))(
        interval_midpoints
    )
    left_states = jax.tree.map(lambda value: value[:-1], solution.states)
    right_states = jax.tree.map(lambda value: value[1:], solution.states)
    left_heat = experiment.prepared_model.evaluate(
        left_states, parameters, interval_currents
    ).generated_heat_w_m3
    right_heat = experiment.prepared_model.evaluate(
        right_states, parameters, interval_currents
    ).generated_heat_w_m3
    cell_volume = 0.1 * 2.5e-4
    expected_generated_energy = jnp.sum(
        0.5 * interval_duration * (left_heat + right_heat) * cell_volume
    )
    np.testing.assert_allclose(
        result.ledger.integrated_generated_heat_j,
        expected_generated_energy,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    observation_policy = protocol.input_policy(protocol_values)
    node_currents = jax.vmap(
        lambda time: observation_policy.evaluate(time, jnp.asarray(0.0))[0]
    )(solution.times)
    node_heat = experiment.prepared_model.evaluate(
        solution.states, parameters, node_currents
    ).generated_heat_w_m3
    cross_jump_trapezoid = jnp.sum(
        0.5 * interval_duration * (node_heat[:-1] + node_heat[1:]) * cell_volume
    )
    assert float(jnp.abs(expected_generated_energy - cross_jump_trapezoid)) > 0.0
    assert int(result.application_status) == int(BatteryRunStatus.SUCCESS)
    assert bool(result.ledger.electrochemical.solid_lithium_conserved)
    assert bool(result.ledger.electrochemical.electrolyte_lithium_conserved)
    assert bool(result.ledger.electrochemical.current_conserved)
    assert bool(result.ledger.electrochemical.interfaces_conserved)
    assert bool(result.ledger.electrochemical.current_split_conserved)
    assert bool(result.ledger.thermal_energy_balanced)
    assert bool(result.ledger.property_support_valid)
    assert bool(result.ledger.applicability_conditions_satisfied)
    assert bool(result.ledger.domain_valid)
    assert bool(result.ledger.finite)
    assert bool(result.ledger.successful)
    assert bool(jnp.all(result.outputs.valid))
    np.testing.assert_allclose(
        result.ledger.maximum_heat_sum_residual_w_m3, 0.0, atol=1.0e-8
    )
    np.testing.assert_allclose(
        result.ledger.maximum_power_identity_residual_w_m3, 0.0, atol=1.0e-8
    )

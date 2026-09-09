#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
from phydrax.applications.battery._qualification import (
    MARQUIS_2019_SPME_CANDIDATE,
    MARQUIS_2019_SPME_SUPPORT,
)
from phydrax.applications.battery._results import BatteryRunStatus
from phydrax.applications.battery._spm import (
    PrescribedCurrentSpmAdapter,
    SpmInitialCondition,
    SpmParameters,
)
from phydrax.applications.battery._spme_marquis2019 import (
    _eq49_ocp_error,
    _ocp_second_derivative_concentration,
    Marquis2019SpmeAdapter,
    Marquis2019SpmeInitialCondition,
    Marquis2019SpmeParameters,
    Marquis2019SpmePlan,
    Marquis2019SpmeState,
    validate_marquis2019_spme_execution,
)
from phydrax.applications.battery._through_cell import (
    PreparedThroughCellMesh,
    ThroughCellRegionPlan,
)


FARADAY = 96485.33212
GAS_CONSTANT = 8.31446261815324


def _constant(value, support, *, quantity, value_unit):
    return ConstantPropertyLaw(
        jnp.asarray(value),
        jnp.asarray(support),
        value_bounds=(0.0, jnp.inf),
        quantity=quantity,
        coordinate="temperature" if value_unit != "V" else "stoichiometry",
        value_unit=value_unit,
        coordinate_unit="K" if value_unit != "V" else "1",
        source_id=f"test:{quantity}:{value}",
    )


def _electrolyte_law(value, *, quantity, value_unit):
    bounds = (0.0, 1.0) if value_unit == "1" else (0.0, jnp.inf)
    return ConcentrationTemperaturePropertyLaw(
        jnp.asarray((100.0, 2000.0)),
        jnp.asarray((250.0, 350.0)),
        jnp.full((2, 2), value),
        value_bounds=bounds,
        quantity=quantity,
        value_unit=value_unit,
        source_id=f"test:{quantity}:{value}",
    )


def _spm_parameters(*, maximum_current=20.0):
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
        temperature_k=298.15,
        maximum_absolute_current_a=maximum_current,
        negative_stoichiometry_at_empty=0.1,
        negative_stoichiometry_at_full=0.9,
        positive_stoichiometry_at_empty=0.9,
        positive_stoichiometry_at_full=0.1,
        negative_solid_diffusivity=_constant(
            2.0e-14,
            (250.0, 350.0),
            quantity="negative-solid-diffusivity",
            value_unit="m2/s",
        ),
        positive_solid_diffusivity=_constant(
            1.0e-14,
            (250.0, 350.0),
            quantity="positive-solid-diffusivity",
            value_unit="m2/s",
        ),
        negative_exchange_current_density=_constant(
            5.0,
            (250.0, 350.0),
            quantity="negative-exchange-current",
            value_unit="A/m2",
        ),
        positive_exchange_current_density=_constant(
            4.0,
            (250.0, 350.0),
            quantity="positive-exchange-current",
            value_unit="A/m2",
        ),
        negative_open_circuit_potential=_constant(
            0.1,
            (0.0, 1.0),
            quantity="negative-ocp",
            value_unit="V",
        ),
        positive_open_circuit_potential=_constant(
            4.1,
            (0.0, 1.0),
            quantity="positive-ocp",
            value_unit="V",
        ),
    )


def _parameters(*, maximum_current=20.0, transference=0.4):
    return Marquis2019SpmeParameters(
        _spm_parameters(maximum_current=maximum_current),
        separator_thickness_m=5.0e-5,
        negative_electrolyte_porosity=0.3,
        separator_electrolyte_porosity=0.5,
        positive_electrolyte_porosity=0.3,
        bruggeman_coefficient=1.5,
        typical_electrolyte_concentration_mol_m3=1000.0,
        electrolyte_diffusivity=_electrolyte_law(
            2.0e-10,
            quantity="electrolyte-diffusivity",
            value_unit="m2/s",
        ),
        electrolyte_conductivity=_electrolyte_law(
            1.1,
            quantity="electrolyte-conductivity",
            value_unit="S/m",
        ),
        transference_number=_electrolyte_law(
            transference,
            quantity="transference-number",
            value_unit="1",
        ),
        negative_solid_conductivity_s_m=100.0,
        positive_solid_conductivity_s_m=80.0,
    )


def _adapter(*, shells=5, negative_cells=4, separator_cells=3, positive_cells=4):
    adapter = Marquis2019SpmeAdapter(
        Marquis2019SpmePlan(
            shells,
            negative_electrolyte_cell_count=negative_cells,
            separator_electrolyte_cell_count=separator_cells,
            positive_electrolyte_cell_count=positive_cells,
        )
    )
    return adapter, adapter.prepare()


def _runtime(parameters, current, *, duration=2.0):
    protocol = BatteryProtocolPlan((CurrentStepPlan(duration),))
    values = BatteryProtocolValues(protocol, jnp.asarray((current,)))
    return BatteryRuntimeInputs(
        parameters,
        protocol.input_policy(values, node_side="right"),
        values.stop_thresholds,
        protocol_id=protocol.protocol_id,
        observation_input_policy=protocol.input_policy(values),
    )


def _state(adapter, prepared, parameters, *, negative=0.5, positive=0.5):
    return adapter.initial_state(
        prepared,
        parameters,
        Marquis2019SpmeInitialCondition(negative, positive),
    )


def _observable(adapter, output, name):
    return output.values[..., adapter.observable_names.index(name)]


def test_three_region_plan_prepares_exact_ordered_topology_and_nonuniform_faces():
    negative = ThroughCellRegionPlan(
        2, region="negative", reference_faces=(0.0, 0.25, 1.0)
    )
    separator = ThroughCellRegionPlan(1, region="separator")
    positive = ThroughCellRegionPlan(
        2, region="positive", reference_faces=(0.0, 0.75, 1.0)
    )
    prepared = PreparedThroughCellMesh(negative, separator, positive)

    assert prepared.cell_count == 5
    assert prepared.negative_separator_face_index == 2
    assert prepared.separator_positive_face_index == 3
    np.testing.assert_array_equal(
        prepared.cell_region_indices, jnp.asarray((0, 0, 1, 2, 2))
    )
    np.testing.assert_allclose(
        prepared.electrolyte_current_fraction_faces,
        jnp.asarray((0.0, 0.25, 1.0, 1.0, 0.25, 0.0)),
    )


def test_eq48_boundaries_interfaces_current_split_source_and_conservation():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    terminal_current = -6.0
    evaluation = prepared.evaluate(state, parameters, terminal_current)
    electrolyte = evaluation.electrolyte_transport
    paper_current_density = (
        -terminal_current / parameters.spm_parameters.electrode_area_m2
    )
    ns_face = prepared.through_cell.negative_separator_face_index
    sp_face = prepared.through_cell.separator_positive_face_index

    np.testing.assert_allclose(
        electrolyte.total_molar_flux_mol_m2_s[..., (0, -1)], 0.0, atol=1.0e-16
    )
    np.testing.assert_allclose(
        electrolyte.total_molar_flux_mol_m2_s[..., (ns_face, sp_face)],
        parameters.transference_number.evaluate(
            parameters.typical_electrolyte_concentration_mol_m3,
            parameters.spm_parameters.temperature_k,
        ).values
        * paper_current_density
        / FARADAY,
    )
    np.testing.assert_allclose(
        electrolyte.electrolyte_current_density_a_m2[..., (ns_face, sp_face)],
        paper_current_density,
    )
    np.testing.assert_allclose(
        electrolyte.electrolyte_current_density_a_m2
        + electrolyte.solid_current_density_a_m2,
        paper_current_density,
    )
    np.testing.assert_allclose(
        electrolyte.negative_separator_concentration_jump_mol_m3,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        electrolyte.separator_positive_concentration_jump_mol_m3,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        electrolyte.source_mol_m3_s[prepared.through_cell.separator_mask], 0.0
    )
    assert bool(
        jnp.all(electrolyte.source_mol_m3_s[prepared.through_cell.negative_mask] > 0.0)
    )
    assert bool(
        jnp.all(electrolyte.source_mol_m3_s[prepared.through_cell.positive_mask] < 0.0)
    )
    np.testing.assert_allclose(jnp.sum(electrolyte.amount_rate_mol_s), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        electrolyte.equation_mapping_residual_mol_s, 0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        jnp.sum(evaluation.particle_transport.negative.amount_rate_mol_s),
        terminal_current / FARADAY,
    )
    np.testing.assert_allclose(
        jnp.sum(evaluation.particle_transport.positive.amount_rate_mol_s),
        -terminal_current / FARADAY,
    )
    np.testing.assert_allclose(
        jnp.sum(evaluation.particle_transport.negative.amount_rate_mol_s)
        + jnp.sum(evaluation.particle_transport.positive.amount_rate_mol_s),
        0.0,
        atol=1.0e-15,
    )


def test_single_face_flux_preserves_continuity_and_inventory_for_nonuniform_state():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    metrics = prepared.through_cell.metrics(
        negative_thickness_m=parameters.spm_parameters.negative_electrode_thickness_m,
        separator_thickness_m=parameters.separator_thickness_m,
        positive_thickness_m=parameters.spm_parameters.positive_electrode_thickness_m,
        negative_porosity=parameters.negative_electrolyte_porosity,
        separator_porosity=parameters.separator_electrolyte_porosity,
        positive_porosity=parameters.positive_electrolyte_porosity,
        bruggeman_coefficient=parameters.bruggeman_coefficient,
        electrolyte_diffusivity_m2_s=parameters.electrolyte_diffusivity.evaluate(
            parameters.typical_electrolyte_concentration_mol_m3,
            parameters.spm_parameters.temperature_k,
        ).values,
        electrode_area_m2=parameters.spm_parameters.electrode_area_m2,
    )
    concentration = 900.0 + 2.0e6 * metrics.cell_centers_m
    perturbed = Marquis2019SpmeState(
        state.negative_amount_mol,
        state.positive_amount_mol,
        concentration * metrics.storage_volume_m3,
    )
    electrolyte = prepared.evaluate(perturbed, parameters, 0.0).electrolyte_transport

    np.testing.assert_allclose(
        electrolyte.negative_separator_concentration_jump_mol_m3,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        electrolyte.separator_positive_concentration_jump_mol_m3,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(jnp.sum(electrolyte.amount_rate_mol_s), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(electrolyte.conservation_residual_mol_s, 0.0, atol=1.0e-15)


def test_electrolyte_coefficients_are_frozen_at_typical_concentration_and_temperature():
    parameters = _parameters()
    concentration_nodes = jnp.asarray((500.0, 1000.0, 1500.0))
    temperature_nodes = jnp.asarray((250.0, 350.0))

    def law(values, *, quantity, value_unit, bounds):
        return ConcentrationTemperaturePropertyLaw(
            concentration_nodes,
            temperature_nodes,
            jnp.asarray(values),
            value_bounds=bounds,
            quantity=quantity,
            value_unit=value_unit,
            source_id=f"test:variable:{quantity}",
        )

    diffusivity = law(
        (
            (1.0e-10, 1.5e-10),
            (2.0e-10, 3.0e-10),
            (4.0e-10, 6.0e-10),
        ),
        quantity="electrolyte-diffusivity",
        value_unit="m2/s",
        bounds=(0.0, jnp.inf),
    )
    conductivity = law(
        ((0.5, 0.75), (1.1, 1.65), (2.0, 3.0)),
        quantity="electrolyte-conductivity",
        value_unit="S/m",
        bounds=(0.0, jnp.inf),
    )
    transference = law(
        ((0.2, 0.25), (0.4, 0.45), (0.6, 0.65)),
        quantity="transference-number",
        value_unit="1",
        bounds=(0.0, 1.0),
    )
    variable_parameters = eqx.tree_at(
        lambda value: (
            value.electrolyte_diffusivity,
            value.electrolyte_conductivity,
            value.transference_number,
        ),
        parameters,
        replace=(diffusivity, conductivity, transference),
    )
    adapter, prepared = _adapter()
    uniform = _state(adapter, prepared, variable_parameters)
    current = 0.2
    uniform_evaluation = prepared.evaluate(
        uniform,
        variable_parameters,
        current,
    )
    typical_concentration = variable_parameters.typical_electrolyte_concentration_mol_m3
    temperature = variable_parameters.spm_parameters.temperature_k
    typical_diffusivity = diffusivity.evaluate(
        typical_concentration,
        temperature,
    ).values
    metrics = prepared.through_cell.metrics(
        negative_thickness_m=parameters.spm_parameters.negative_electrode_thickness_m,
        separator_thickness_m=parameters.separator_thickness_m,
        positive_thickness_m=parameters.spm_parameters.positive_electrode_thickness_m,
        negative_porosity=parameters.negative_electrolyte_porosity,
        separator_porosity=parameters.separator_electrolyte_porosity,
        positive_porosity=parameters.positive_electrolyte_porosity,
        bruggeman_coefficient=parameters.bruggeman_coefficient,
        electrolyte_diffusivity_m2_s=typical_diffusivity,
        electrode_area_m2=parameters.spm_parameters.electrode_area_m2,
    )
    local_concentration = jnp.linspace(
        700.0,
        1300.0,
        prepared.through_cell.cell_count,
    )
    evolved = Marquis2019SpmeState(
        uniform.negative_amount_mol,
        uniform.positive_amount_mol,
        local_concentration * metrics.storage_volume_m3,
    )
    evolved_evaluation = prepared.evaluate(
        evolved,
        variable_parameters,
        current,
    )
    expected_conductivity = conductivity.evaluate(
        typical_concentration,
        temperature,
    ).values
    expected_transference = transference.evaluate(
        typical_concentration,
        temperature,
    ).values

    for evaluation in (uniform_evaluation, evolved_evaluation):
        np.testing.assert_allclose(
            evaluation.electrolyte_diffusivity_m2_s,
            typical_diffusivity,
        )
        np.testing.assert_allclose(
            evaluation.electrolyte_conductivity_s_m,
            expected_conductivity,
        )
        np.testing.assert_allclose(
            evaluation.transference_number_value,
            expected_transference,
        )
    np.testing.assert_allclose(
        evolved_evaluation.electrolyte_ohmic_loss_v,
        uniform_evaluation.electrolyte_ohmic_loss_v,
    )

    shifted_typical = eqx.tree_at(
        lambda value: value.typical_electrolyte_concentration_mol_m3,
        variable_parameters,
        jnp.asarray(1200.0),
    )
    shifted_evaluation = prepared.evaluate(evolved, shifted_typical, current)
    assert not bool(
        jnp.isclose(
            shifted_evaluation.electrolyte_diffusivity_m2_s,
            typical_diffusivity,
            rtol=1.0e-12,
            atol=0.0,
        )
    )
    hot_spm = eqx.tree_at(
        lambda value: value.temperature_k,
        variable_parameters.spm_parameters,
        jnp.asarray(320.0),
    )
    hot_parameters = eqx.tree_at(
        lambda value: value.spm_parameters,
        variable_parameters,
        hot_spm,
    )
    hot_evaluation = prepared.evaluate(evolved, hot_parameters, current)
    assert not bool(
        jnp.isclose(
            hot_evaluation.electrolyte_diffusivity_m2_s,
            typical_diffusivity,
            rtol=1.0e-12,
            atol=0.0,
        )
    )
    total_length = (
        parameters.spm_parameters.negative_electrode_thickness_m
        + parameters.separator_thickness_m
        + parameters.spm_parameters.positive_electrode_thickness_m
    )
    expected_ce = (
        abs(current)
        / parameters.spm_parameters.electrode_area_m2
        * total_length
        / (
            typical_diffusivity
            * FARADAY
            * parameters.spm_parameters.negative_maximum_concentration_mol_m3
        )
    )
    np.testing.assert_allclose(
        evolved_evaluation.electrolyte_migration_number,
        expected_ce,
    )


def test_five_voltage_terms_are_exact_paper_averages_with_one_third_factors():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    current = 4.0
    output = adapter.observe(
        prepared, jnp.asarray(0.0), state, _runtime(parameters, current)
    )
    terms = tuple(
        _observable(adapter, output, name)
        for name in (
            "voltage:particle_ocp_v",
            "voltage:reaction_overpotential_v",
            "voltage:mean_concentration_overpotential_v",
            "voltage:electrolyte_ohmic_loss_v",
            "voltage:solid_ohmic_loss_v",
        )
    )
    spm = parameters.spm_parameters
    current_density = current / spm.electrode_area_m2
    expected_electrolyte_ohmic = (
        current_density
        / (
            parameters.electrolyte_conductivity.evaluate(
                parameters.typical_electrolyte_concentration_mol_m3,
                spm.temperature_k,
            ).values
        )
        * (
            spm.negative_electrode_thickness_m
            / (
                3.0
                * parameters.negative_electrolyte_porosity
                ** parameters.bruggeman_coefficient
            )
            + parameters.separator_thickness_m
            / parameters.separator_electrolyte_porosity**parameters.bruggeman_coefficient
            + spm.positive_electrode_thickness_m
            / (
                3.0
                * parameters.positive_electrolyte_porosity
                ** parameters.bruggeman_coefficient
            )
        )
    )
    expected_solid_ohmic = (
        current_density
        / 3.0
        * (
            spm.negative_electrode_thickness_m
            / parameters.negative_solid_conductivity_s_m
            + spm.positive_electrode_thickness_m
            / parameters.positive_solid_conductivity_s_m
        )
    )

    np.testing.assert_allclose(terms[0], 4.0)
    assert float(terms[1]) > 0.0
    np.testing.assert_allclose(terms[2], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(terms[3], expected_electrolyte_ohmic)
    np.testing.assert_allclose(terms[4], expected_solid_ohmic)
    np.testing.assert_allclose(_observable(adapter, output, "voltage_v"), sum(terms))
    assert bool(output.domain_valid)


def test_uniform_electrolyte_and_vanishing_ohmic_losses_reduce_exactly_to_spm():
    parameters = _parameters(transference=1.0)
    high_conductivity = _electrolyte_law(
        1.0e30,
        quantity="electrolyte-conductivity",
        value_unit="S/m",
    )
    parameters = eqx.tree_at(
        lambda value: (
            value.electrolyte_conductivity,
            value.negative_solid_conductivity_s_m,
            value.positive_solid_conductivity_s_m,
        ),
        parameters,
        replace=(high_conductivity, jnp.asarray(1.0e30), jnp.asarray(1.0e30)),
    )
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    current = 3.0
    runtime = _runtime(parameters, current)
    spme_rate = adapter.problem(prepared, state, runtime).drift(
        jnp.asarray(0.0), state, runtime
    )

    spm_adapter = PrescribedCurrentSpmAdapter(adapter.plan.spm_plan)
    spm_state = spm_adapter.initial_state(
        prepared.spm,
        parameters.spm_parameters,
        SpmInitialCondition(0.5, 0.5),
    )
    spm_runtime = _runtime(parameters.spm_parameters, current)
    spm_rate = spm_adapter.problem(prepared.spm, spm_state, spm_runtime).drift(
        jnp.asarray(0.0), spm_state, spm_runtime
    )
    spme_output = adapter.observe(prepared, jnp.asarray(0.0), state, runtime)
    spm_output = spm_adapter.observe(
        prepared.spm, jnp.asarray(0.0), spm_state, spm_runtime
    )

    np.testing.assert_allclose(
        spme_rate.negative_amount_mol, spm_rate.negative_amount_mol
    )
    np.testing.assert_allclose(
        spme_rate.positive_amount_mol, spm_rate.positive_amount_mol
    )
    np.testing.assert_allclose(spme_rate.electrolyte_amount_mol, 0.0, atol=1.0e-18)
    np.testing.assert_allclose(
        _observable(adapter, spme_output, "voltage_v"),
        _observable(spm_adapter, spm_output, "voltage_v"),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_eq48_particles_use_runtime_shell_concentration_dependent_diffusivity():
    parameters = _parameters()
    variable_diffusivity = TabulatedPropertyLaw(
        jnp.asarray((0.0, 3.0e4)),
        jnp.asarray((1.0e-14, 5.0e-14)),
        value_bounds=(0.0, jnp.inf),
        quantity="negative-solid-diffusivity",
        coordinate="solid_lithium_concentration",
        value_unit="m2/s",
        coordinate_unit="mol/m3",
        source_id="test:negative-variable-solid-diffusivity",
    )
    variable_spm = eqx.tree_at(
        lambda value: value.negative_solid_diffusivity,
        parameters.spm_parameters,
        variable_diffusivity,
    )
    variable_parameters = eqx.tree_at(
        lambda value: value.spm_parameters,
        parameters,
        variable_spm,
    )
    adapter, prepared = _adapter()
    uniform = _state(adapter, prepared, variable_parameters)
    shell_scale = jnp.linspace(0.8, 1.2, uniform.negative_amount_mol.size)
    state = Marquis2019SpmeState(
        uniform.negative_amount_mol * shell_scale,
        uniform.positive_amount_mol,
        uniform.electrolyte_amount_mol,
    )
    current = 0.1
    variable = prepared.evaluate(state, variable_parameters, current)
    constant = prepared.evaluate(state, parameters, current)
    expected_shell_diffusivity = variable_diffusivity.evaluate(
        variable.particle_transport.negative.concentration_mol_m3
    ).values
    np.testing.assert_allclose(
        variable.particle_transport.negative_diffusivity_m2_s,
        expected_shell_diffusivity,
    )
    assert (
        float(
            jnp.max(
                jnp.abs(
                    variable.particle_transport.negative.amount_rate_mol_s
                    - constant.particle_transport.negative.amount_rate_mol_s
                )
            )
        )
        > 1.0e-12
    )

    spm = variable_parameters.spm_parameters
    total_length = (
        spm.negative_electrode_thickness_m
        + variable_parameters.separator_thickness_m
        + spm.positive_electrode_thickness_m
    )
    expected_table6 = (
        spm.negative_particle_radius_m**2
        * (abs(current) / spm.electrode_area_m2)
        / (
            jnp.min(expected_shell_diffusivity)
            * FARADAY
            * spm.negative_maximum_concentration_mol_m3
            * total_length
        )
    )
    np.testing.assert_allclose(
        variable.negative_solid_diffusion_number,
        expected_table6,
    )


def test_table6_quantities_and_eq49_use_actual_current_and_state():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    current = -0.5
    output = adapter.observe(
        prepared, jnp.asarray(0.0), state, _runtime(parameters, current)
    )
    spm = parameters.spm_parameters
    current_density = abs(current) / spm.electrode_area_m2
    total_length = (
        spm.negative_electrode_thickness_m
        + parameters.separator_thickness_m
        + spm.positive_electrode_thickness_m
    )
    expected_ce = (
        current_density
        * total_length
        / (
            parameters.electrolyte_diffusivity.evaluate(
                parameters.typical_electrolyte_concentration_mol_m3,
                spm.temperature_k,
            ).values
            * FARADAY
            * spm.negative_maximum_concentration_mol_m3
        )
    )
    expected_solid_ratio = (
        GAS_CONSTANT
        * spm.temperature_k
        * parameters.negative_solid_conductivity_s_m
        / (FARADAY * current_density * total_length)
    )

    np.testing.assert_allclose(
        _observable(adapter, output, "table6:electrolyte_migration_number"),
        expected_ce,
    )
    np.testing.assert_allclose(
        _observable(adapter, output, "table6:negative_thermal_to_solid_ohmic_ratio"),
        expected_solid_ratio,
    )
    np.testing.assert_allclose(
        _observable(adapter, output, "eq49:electrolyte_error"), expected_ce**2
    )
    np.testing.assert_allclose(
        _observable(adapter, output, "eq49:negative_ocp_error"), 0.0
    )
    np.testing.assert_allclose(
        _observable(adapter, output, "eq49:positive_ocp_error"), 0.0
    )
    np.testing.assert_allclose(
        _observable(adapter, output, "eq49:error_estimate"), expected_ce**2
    )
    np.testing.assert_allclose(
        _observable(adapter, output, "table6:conditions_satisfied"), 1.0
    )


def test_eq49_ocp_error_matches_independent_molar_perturbation_normalization():
    # The runtime passes q_scale=I*L/De [C/m3], whereas a Taylor expansion
    # uses delta_c=q_scale/F [mol/m3] and dimensionless voltage scale RT/F.
    molar_perturbation = jnp.asarray(2.0)
    charge_concentration_scale = FARADAY * molar_perturbation
    curvature = jnp.asarray(-3.0e-7)
    thermal_energy = jnp.asarray(GAS_CONSTANT * 298.15)
    expected = molar_perturbation**2 * jnp.abs(curvature) / (thermal_energy / FARADAY)
    np.testing.assert_allclose(
        _eq49_ocp_error(charge_concentration_scale, curvature, thermal_energy),
        expected,
        rtol=1.0e-12,
    )


def test_eq49_applicability_refuses_a_tabulated_ocp_kink_in_its_neighborhood():
    parameters = _parameters()
    kinked_ocp = TabulatedPropertyLaw(
        jnp.asarray((0.0, 0.5, 1.0)),
        jnp.asarray((0.1, 0.1, 0.2)),
        quantity="negative-ocp",
        coordinate="stoichiometry",
        value_unit="V",
        coordinate_unit="1",
        source_id="test:negative-ocp-sharp-kink",
    )
    candidate_spm = eqx.tree_at(
        lambda value: value.negative_open_circuit_potential,
        parameters.spm_parameters,
        kinked_ocp,
    )
    candidate = eqx.tree_at(
        lambda value: value.spm_parameters,
        parameters,
        candidate_spm,
    )
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, candidate)
    output = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        state,
        _runtime(candidate, 0.1),
    )

    assert bool(output.domain_valid)
    np.testing.assert_allclose(_observable(adapter, output, "eq49:applicable"), 0.0)
    np.testing.assert_allclose(
        _observable(adapter, output, "table6:conditions_satisfied"), 0.0
    )


def test_eq49_neighborhood_refuses_crossing_noncontiguous_table_support():
    disconnected_ocp = TabulatedPropertyLaw(
        jnp.asarray((0.0, 0.45, 0.5, 0.55, 1.0)),
        jnp.asarray((0.1, 0.145, 0.15, 0.155, 0.2)),
        source_mask=jnp.asarray((True, True, False, True, True)),
        quantity="negative-ocp",
        coordinate="stoichiometry",
        value_unit="V",
        coordinate_unit="1",
        source_id="test:negative-ocp-disconnected-support",
    )
    lower = disconnected_ocp.evaluate(jnp.asarray(0.2))
    upper = disconnected_ocp.evaluate(jnp.asarray(0.6))
    _, neighborhood_valid = _ocp_second_derivative_concentration(
        disconnected_ocp,
        jnp.asarray(0.4),
        jnp.asarray(1.0),
        jnp.asarray(0.2),
    )

    assert bool(lower.support)
    assert bool(upper.support)
    assert not bool(neighborhood_valid)


def test_support_failures_are_explicit_and_outputs_remain_finite():
    parameters = _parameters(maximum_current=5.0)
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    unsupported = adapter.observe(
        prepared, jnp.asarray(0.0), state, _runtime(parameters, 6.0)
    )
    assert not bool(unsupported.domain_valid)
    assert bool(jnp.all(jnp.isfinite(unsupported.values)))
    np.testing.assert_allclose(
        _observable(adapter, unsupported, "current_envelope_margin_a"), -1.0
    )

    invalid_state = Marquis2019SpmeState(
        state.negative_amount_mol,
        state.positive_amount_mol,
        state.electrolyte_amount_mol.at[0].set(-1.0),
    )
    invalid = adapter.observe(
        prepared, jnp.asarray(0.0), invalid_state, _runtime(parameters, 0.0)
    )
    assert not bool(invalid.domain_valid)
    assert bool(jnp.all(jnp.isfinite(invalid.values)))


def test_through_cell_refinement_improves_interface_flux_for_smooth_solution():
    parameters = _parameters()
    parameters = eqx.tree_at(
        lambda value: value.separator_electrolyte_porosity,
        parameters,
        parameters.negative_electrolyte_porosity,
    )
    coarse_adapter, coarse = _adapter(
        negative_cells=2, separator_cells=2, positive_cells=2
    )
    fine_adapter, fine = _adapter(negative_cells=8, separator_cells=8, positive_cells=8)

    def interface_error(adapter, prepared):
        state = _state(adapter, prepared, parameters)
        metrics = prepared.through_cell.metrics(
            negative_thickness_m=parameters.spm_parameters.negative_electrode_thickness_m,
            separator_thickness_m=parameters.separator_thickness_m,
            positive_thickness_m=parameters.spm_parameters.positive_electrode_thickness_m,
            negative_porosity=parameters.negative_electrolyte_porosity,
            separator_porosity=parameters.separator_electrolyte_porosity,
            positive_porosity=parameters.positive_electrolyte_porosity,
            bruggeman_coefficient=parameters.bruggeman_coefficient,
            electrolyte_diffusivity_m2_s=parameters.electrolyte_diffusivity.evaluate(
                parameters.typical_electrolyte_concentration_mol_m3,
                parameters.spm_parameters.temperature_k,
            ).values,
            electrode_area_m2=parameters.spm_parameters.electrode_area_m2,
        )
        alpha = 1000.0
        concentration = 900.0 + 100.0 * jnp.exp(alpha * metrics.cell_centers_m)
        candidate = Marquis2019SpmeState(
            state.negative_amount_mol,
            state.positive_amount_mol,
            concentration * metrics.storage_volume_m3,
        )
        transport = prepared.evaluate(candidate, parameters, 0.0).electrolyte_transport
        face = prepared.through_cell.negative_separator_face_index
        location = parameters.spm_parameters.negative_electrode_thickness_m
        effective_diffusivity = (
            parameters.negative_electrolyte_porosity**parameters.bruggeman_coefficient
            * parameters.electrolyte_diffusivity.evaluate(
                parameters.typical_electrolyte_concentration_mol_m3,
                parameters.spm_parameters.temperature_k,
            ).values
        )
        exact = -effective_diffusivity * 100.0 * alpha * jnp.exp(alpha * location)
        return jnp.abs(transport.diffusive_molar_flux_mol_m2_s[face] - exact)

    assert float(interface_error(fine_adapter, fine)) < float(
        interface_error(coarse_adapter, coarse)
    )


def test_problem_observation_jit_vmap_and_fixed_path_derivatives():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    runtime = _runtime(parameters, 3.0)
    problem = adapter.problem(prepared, state, runtime)
    eager_rate = problem.drift(jnp.asarray(0.0), state, runtime)
    compiled_rate = jax.jit(
        lambda time, candidate: problem.drift(time, candidate, runtime)
    )(jnp.asarray(0.0), state)
    np.testing.assert_allclose(
        compiled_rate.negative_amount_mol, eager_rate.negative_amount_mol
    )
    np.testing.assert_allclose(
        compiled_rate.electrolyte_amount_mol,
        eager_rate.electrolyte_amount_mol,
        atol=1.0e-18,
    )
    compiled_output = jax.jit(
        lambda time, candidate: adapter.observe(prepared, time, candidate, runtime)
    )(jnp.asarray(0.0), state)
    assert bool(compiled_output.domain_valid)

    states = jax.tree.map(lambda value: jnp.stack((value, value)), state)
    mapped = jax.vmap(
        lambda one_state: adapter.observe(prepared, jnp.asarray(0.0), one_state, runtime)
    )(states)
    assert mapped.values.shape == (2, len(adapter.observable_names))
    assert bool(jnp.all(mapped.domain_valid))

    def voltage_for_current(current):
        output = adapter.observe(
            prepared,
            jnp.asarray(0.0),
            state,
            _runtime(parameters, current),
        )
        return _observable(adapter, output, "voltage_v")

    current_derivative = jax.grad(voltage_for_current)(jnp.asarray(1.0))
    assert bool(jnp.isfinite(current_derivative))
    assert float(current_derivative) > 0.0

    def voltage_for_conductivity(conductivity):
        candidate_conductivity = eqx.tree_at(
            lambda law: law.values,
            parameters.electrolyte_conductivity,
            jnp.zeros_like(parameters.electrolyte_conductivity.values) + conductivity,
        )
        candidate = eqx.tree_at(
            lambda value: value.electrolyte_conductivity,
            parameters,
            candidate_conductivity,
        )
        output = adapter.observe(
            prepared,
            jnp.asarray(0.0),
            state,
            _runtime(candidate, 2.0),
        )
        return _observable(adapter, output, "voltage_v")

    conductivity_derivative = jax.grad(voltage_for_conductivity)(jnp.asarray(1.1))
    assert bool(jnp.isfinite(conductivity_derivative))
    assert float(conductivity_derivative) < 0.0


def test_protocol_orchestration_reports_all_conservation_and_evidence():
    parameters = _parameters()
    adapter = Marquis2019SpmeAdapter(
        Marquis2019SpmePlan(
            5,
            negative_electrolyte_cell_count=3,
            separator_electrolyte_cell_count=2,
            positive_electrolyte_cell_count=3,
        )
    )
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(0.5), RestStepPlan(0.25), CurrentStepPlan(0.25))
    )
    protocol_values = BatteryProtocolValues(protocol, jnp.asarray((0.2, -0.05)))
    support = MARQUIS_2019_SPME_SUPPORT
    profile = MARQUIS_2019_SPME_CANDIDATE
    save_times = jnp.asarray((0.0, 0.25, 0.5, 0.75, 1.0))
    experiment = BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(
            (
                "voltage_v",
                "current_a",
                "total_lithium_mol",
                "eq49:error_estimate",
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
        Marquis2019SpmeInitialCondition(0.5, 0.5),
        protocol_values,
    )

    assert int(result.application_status) == int(BatteryRunStatus.SUCCESS)
    assert bool(result.ledger.solid_lithium_conserved)
    assert bool(result.ledger.electrolyte_lithium_conserved)
    assert bool(result.ledger.total_lithium_conserved)
    assert bool(result.ledger.current_conserved)
    assert bool(result.ledger.interfaces_conserved)
    assert bool(result.ledger.current_split_conserved)
    assert bool(result.ledger.eq49_applicable)
    assert bool(result.ledger.asymptotic_conditions_satisfied)
    assert bool(result.ledger.domain_valid)
    assert bool(result.ledger.successful)
    np.testing.assert_allclose(
        result.ledger.integrated_terminal_charge_c, 0.0875, rtol=1.0e-12
    )
    assert bool(jnp.all(result.outputs.valid))

    def final_negative_amount(first_current):
        dynamic_values = BatteryProtocolValues(
            protocol,
            jnp.stack((first_current, jnp.asarray(-0.05))),
        )
        dynamic_result = experiment.run(
            parameters,
            Marquis2019SpmeInitialCondition(0.5, 0.5),
            dynamic_values,
        )
        return jnp.sum(dynamic_result.native_solution.states.negative_amount_mol[-1])

    fixed_path_derivative = jax.grad(final_negative_amount)(jnp.asarray(0.2))
    np.testing.assert_allclose(fixed_path_derivative, 0.5 / FARADAY, rtol=2.0e-5)


def test_model_preflight_checks_later_amplitudes_and_complete_charge_inventory():
    parameters = _parameters(maximum_current=0.5)
    _, prepared = _adapter()
    initial = Marquis2019SpmeInitialCondition(0.5, 0.5)
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(0.5), RestStepPlan(0.25), CurrentStepPlan(0.25))
    )
    validate_marquis2019_spme_execution(
        prepared,
        parameters,
        initial,
        protocol,
        BatteryProtocolValues(protocol, jnp.asarray((0.2, -0.05))),
    )
    with pytest.raises(ValueError):
        validate_marquis2019_spme_execution(
            prepared,
            parameters,
            initial,
            protocol,
            BatteryProtocolValues(protocol, jnp.asarray((0.2, 0.6))),
        )
    too_long = BatteryProtocolPlan((CurrentStepPlan(200000.0),))
    with pytest.raises(ValueError):
        validate_marquis2019_spme_execution(
            prepared,
            parameters,
            initial,
            too_long,
            BatteryProtocolValues(too_long, jnp.asarray((0.2,))),
        )


def test_model_preflight_refuses_table6_failure_below_declared_current_bound():
    parameters = _parameters(maximum_current=20.0)
    _, prepared = _adapter()
    protocol = BatteryProtocolPlan((CurrentStepPlan(0.1),))
    with pytest.raises(ValueError):
        validate_marquis2019_spme_execution(
            prepared,
            parameters,
            Marquis2019SpmeInitialCondition(0.5, 0.5),
            protocol,
            BatteryProtocolValues(protocol, jnp.asarray((5.0,))),
        )

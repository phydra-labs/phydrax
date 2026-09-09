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
    Marquis2019SpmeAdapter,
    Marquis2019SpmeParameters,
    Marquis2019SpmeState,
)
from phydrax.applications.battery._spme_side_reactions import (
    BrosaPlanellaSpmeSeiAdapter,
    BrosaPlanellaSpmeSeiInitialCondition,
    BrosaPlanellaSpmeSeiParameters,
    BrosaPlanellaSpmeSeiPlan,
    BrosaPlanellaSpmeSeiState,
)
from phydrax.qualification import CapabilityProfile, SupportTuple


FARADAY = 96485.33212


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


def _marquis_parameters(*, maximum_current=20.0):
    return Marquis2019SpmeParameters(
        _spm_parameters(maximum_current=maximum_current),
        separator_thickness_m=5.0e-5,
        negative_electrolyte_porosity=0.3,
        separator_electrolyte_porosity=0.5,
        positive_electrolyte_porosity=0.3,
        bruggeman_coefficient=1.5,
        typical_electrolyte_concentration_mol_m3=1000.0,
        electrolyte_diffusivity=ConcentrationTemperaturePropertyLaw(
            jnp.asarray((100.0, 1000.0, 3000.0)),
            jnp.asarray((280.0, 320.0)),
            jnp.asarray(
                (
                    (1.5e-10, 1.5e-10),
                    (2.0e-10, 2.0e-10),
                    (3.0e-10, 3.0e-10),
                )
            ),
            value_bounds=(0.0, jnp.inf),
            quantity="electrolyte-diffusivity",
            value_unit="m2/s",
            source_id="test:sei-electrolyte-diffusivity",
        ),
        electrolyte_conductivity=ConcentrationTemperaturePropertyLaw(
            jnp.asarray((100.0, 1000.0, 3000.0)),
            jnp.asarray((280.0, 320.0)),
            jnp.asarray(((0.8, 0.8), (1.1, 1.1), (1.5, 1.5))),
            value_bounds=(0.0, jnp.inf),
            quantity="electrolyte-conductivity",
            value_unit="S/m",
            source_id="test:sei-electrolyte-conductivity",
        ),
        transference_number=ConcentrationTemperaturePropertyLaw(
            jnp.asarray((100.0, 1000.0, 3000.0)),
            jnp.asarray((280.0, 320.0)),
            jnp.asarray(((0.3, 0.3), (0.4, 0.4), (0.5, 0.5))),
            value_bounds=(0.0, 1.0),
            quantity="transference-number",
            value_unit="1",
            source_id="test:sei-transference-number",
        ),
        negative_solid_conductivity_s_m=100.0,
        positive_solid_conductivity_s_m=80.0,
    )


def _parameters(
    *,
    rate=1.0e-14,
    initial_film=1.0e-9,
    conductivity=1.0e-6,
    maximum_current=20.0,
):
    return BrosaPlanellaSpmeSeiParameters(
        _marquis_parameters(maximum_current=maximum_current),
        sei_reaction_rate_m_s=rate,
        sei_solvent_concentration_mol_m3=1000.0,
        sei_solvent_diffusivity_m2_s=2.0e-18,
        sei_transfer_coefficient=0.5,
        sei_open_circuit_potential_v=0.4,
        sei_molar_mass_kg_mol=0.162,
        sei_density_kg_m3=1690.0,
        sei_electron_stoichiometry=2.0,
        sei_conductivity_s_m=conductivity,
        electrolyte_thermodynamic_factor=TabulatedPropertyLaw(
            jnp.asarray((100.0, 1000.0, 3000.0)),
            jnp.asarray((1.1, 1.2, 1.4)),
            value_bounds=(0.0, jnp.inf),
            quantity="electrolyte-thermodynamic-factor",
            coordinate="electrolyte_concentration",
            value_unit="1",
            coordinate_unit="mol/m3",
            source_id="test:sei-electrolyte-thermodynamic-factor",
        ),
        initial_sei_film_thickness_m=initial_film,
    )


def _adapter(
    *,
    shells=5,
    negative_cells=4,
    separator_cells=3,
    positive_cells=4,
    weak_threshold=0.1,
    overpotential_threshold=0.1,
):
    adapter = BrosaPlanellaSpmeSeiAdapter(
        BrosaPlanellaSpmeSeiPlan(
            shells,
            negative_electrolyte_cell_count=negative_cells,
            separator_electrolyte_cell_count=separator_cells,
            positive_electrolyte_cell_count=positive_cells,
            weak_side_reaction_threshold=weak_threshold,
            small_overpotential_threshold_v=overpotential_threshold,
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


def _state(adapter, prepared, parameters):
    return adapter.initial_state(
        prepared,
        parameters,
        BrosaPlanellaSpmeSeiInitialCondition(0.5, 0.5),
    )


def _observable(adapter, output, name):
    return output.values[..., adapter.observable_names.index(name)]


def test_local_side_current_uses_electrode_average_only_at_particle_boundary_and_rest_cancels():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    nonuniform_porosity = state.negative_porosity - jnp.asarray(
        (0.0, 1.0e-4, 2.0e-4, 3.0e-4)
    )
    state = eqx.tree_at(
        lambda value: value.negative_porosity,
        state,
        nonuniform_porosity,
    )
    evaluation = prepared.evaluate(state, parameters, 0.0)

    assert evaluation.sei_current_density_a_m3.shape == (4,)
    assert float(jnp.ptp(evaluation.sei_current_density_a_m3)) > 0.0
    widths = prepared.through_cell.reference_cell_widths[:4]
    expected_average = jnp.sum(evaluation.sei_current_density_a_m3 * widths) / jnp.sum(
        widths
    )
    np.testing.assert_allclose(
        evaluation.electrode_averaged_sei_current_density_a_m3,
        expected_average,
        rtol=1.0e-7,
    )
    expected_side_current = (
        -parameters.spme_parameters.spm_parameters.electrode_area_m2
        * parameters.spme_parameters.spm_parameters.negative_electrode_thickness_m
        * expected_average
    )
    np.testing.assert_allclose(evaluation.side_current_a, expected_side_current)
    np.testing.assert_allclose(
        evaluation.negative_intercalation_current_a,
        -evaluation.side_current_a,
    )
    np.testing.assert_allclose(evaluation.terminal_current_balance_residual_a, 0.0)
    expected_negative_rate = -evaluation.side_current_a / FARADAY
    np.testing.assert_allclose(
        jnp.sum(evaluation.negative_particle_transport.amount_rate_mol_s),
        expected_negative_rate,
        rtol=2.0e-6,
        atol=1.0e-15,
    )


def test_porosity_changes_storage_effective_transport_and_local_film_voltage():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    fresh = prepared.evaluate(state, parameters, 2.0)
    runtime = _runtime(parameters, 2.0)
    source_rate = adapter.problem(prepared, state, runtime).drift(
        jnp.asarray(0.0), state, runtime
    )
    assert bool(jnp.all(source_rate.negative_porosity < 0.0))
    assert float(jnp.ptp(source_rate.negative_porosity)) > 0.0
    source_advanced = eqx.tree_at(
        lambda value: value.negative_porosity,
        state,
        state.negative_porosity + 20.0 * source_rate.negative_porosity,
    )
    source_aged = prepared.evaluate(source_advanced, parameters, 2.0)
    assert (
        float(
            jnp.max(
                jnp.abs(
                    source_aged.electrolyte_transport.concentration_mol_m3
                    - fresh.electrolyte_transport.concentration_mol_m3
                )
            )
        )
        > 0.0
    )
    assert float(source_aged.electrolyte_ohmic_loss_v) != float(
        fresh.electrolyte_ohmic_loss_v
    )
    changed = eqx.tree_at(
        lambda value: value.negative_porosity,
        state,
        state.negative_porosity - jnp.asarray((0.01, 0.02, 0.03, 0.04)),
    )
    aged = prepared.evaluate(changed, parameters, 2.0)

    assert bool(
        jnp.all(
            aged.electrolyte_transport.storage_volume_m3[:4]
            < fresh.electrolyte_transport.storage_volume_m3[:4]
        )
    )
    assert bool(
        jnp.all(
            aged.electrolyte_transport.effective_diffusivity_m2_s[:4]
            < fresh.electrolyte_transport.effective_diffusivity_m2_s[:4]
        )
    )
    temperature = parameters.spme_parameters.spm_parameters.temperature_k
    fresh_conductivity = parameters.spme_parameters.electrolyte_conductivity.evaluate(
        fresh.electrolyte_transport.concentration_mol_m3,
        temperature,
    )
    aged_conductivity = parameters.spme_parameters.electrolyte_conductivity.evaluate(
        aged.electrolyte_transport.concentration_mol_m3,
        temperature,
    )
    assert bool(jnp.any(aged_conductivity.values[:4] != fresh_conductivity.values[:4]))
    assert bool(
        jnp.any(
            aged.electrolyte_transport.electrolyte_diffusivity_m2_s[:4]
            != fresh.electrolyte_transport.electrolyte_diffusivity_m2_s[:4]
        )
    )
    assert bool(
        jnp.any(
            aged.electrolyte_transport.transference_number[:4]
            != fresh.electrolyte_transport.transference_number[:4]
        )
    )
    assert bool(jnp.all(aged.film_thickness_m > fresh.film_thickness_m))
    assert float(aged.film_voltage_correction_v) > float(fresh.film_voltage_correction_v)
    assert float(aged.electrolyte_ohmic_loss_v) > float(fresh.electrolyte_ohmic_loss_v)
    output = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        changed,
        _runtime(parameters, 2.0),
    )
    np.testing.assert_allclose(
        _observable(adapter, output, "voltage:sei_film_correction_v"),
        aged.film_voltage_correction_v,
    )


def test_zero_sei_is_exact_marquis_reduction_for_state_rate_and_observables():
    parameters = _parameters(rate=0.0, initial_film=0.0)
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    runtime = _runtime(parameters, 1.5)
    rate = adapter.problem(prepared, state, runtime).drift(
        jnp.asarray(0.0), state, runtime
    )

    marquis_plan = prepared.plan.marquis_plan
    marquis_adapter = Marquis2019SpmeAdapter(marquis_plan)
    marquis_prepared = prepared.marquis
    marquis_parameters = parameters.spme_parameters
    marquis_state = Marquis2019SpmeState(
        state.negative_amount_mol,
        state.positive_amount_mol,
        state.electrolyte_amount_mol,
    )
    marquis_runtime = _runtime(marquis_parameters, 1.5)
    marquis_rate = marquis_adapter.problem(
        marquis_prepared, marquis_state, marquis_runtime
    ).drift(jnp.asarray(0.0), marquis_state, marquis_runtime)
    np.testing.assert_array_equal(
        rate.negative_amount_mol, marquis_rate.negative_amount_mol
    )
    np.testing.assert_array_equal(
        rate.positive_amount_mol, marquis_rate.positive_amount_mol
    )
    np.testing.assert_array_equal(
        rate.electrolyte_amount_mol, marquis_rate.electrolyte_amount_mol
    )
    np.testing.assert_array_equal(rate.negative_porosity, jnp.zeros((4,)))

    output = adapter.observe(prepared, jnp.asarray(0.0), state, runtime)
    marquis_output = marquis_adapter.observe(
        marquis_prepared,
        jnp.asarray(0.0),
        marquis_state,
        marquis_runtime,
    )
    np.testing.assert_array_equal(
        output.values[: len(marquis_adapter.observable_names)],
        marquis_output.values,
    )
    assert bool(_observable(adapter, output, "spme_sr:zero_sei_reduction"))


def test_domain_and_two_spme_sr_asymptotic_assumptions_fail_explicitly():
    parameters = _parameters(rate=1.0e-11, maximum_current=0.02)
    adapter, prepared = _adapter(weak_threshold=1.0e-5, overpotential_threshold=1.0e-5)
    state = _state(adapter, prepared, parameters)
    evaluation = prepared.evaluate(state, parameters, 0.01)
    assert not bool(evaluation.weak_side_reaction_condition_satisfied)
    assert not bool(evaluation.small_overpotential_condition_satisfied)
    assert not bool(evaluation.asymptotic_conditions_satisfied)
    unsupported = eqx.tree_at(
        lambda value: value.spme_parameters.spm_parameters.temperature_k,
        parameters,
        jnp.asarray(400.0),
    )
    unsupported_evaluation = prepared.evaluate(state, unsupported, 0.0)
    assert not bool(unsupported_evaluation.domain_valid)

    invalid_state = eqx.tree_at(
        lambda value: value.negative_porosity,
        state,
        state.negative_porosity.at[0].set(0.0),
    )
    invalid = prepared.evaluate(invalid_state, parameters, 0.0)
    assert not bool(invalid.domain_valid)
    invalid_output = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        invalid_state,
        _runtime(parameters, 0.0),
    )
    assert not bool(invalid_output.domain_valid)
    assert bool(jnp.all(jnp.isfinite(invalid_output.values)))


def test_jit_vmap_and_gradients_keep_fixed_local_field_shapes():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    runtime = _runtime(parameters, 0.5)
    problem = adapter.problem(prepared, state, runtime)
    eager = problem.drift(jnp.asarray(0.0), state, runtime)
    compiled = jax.jit(
        lambda candidate: problem.drift(jnp.asarray(0.0), candidate, runtime)
    )(state)
    np.testing.assert_allclose(compiled.negative_amount_mol, eager.negative_amount_mol)
    np.testing.assert_allclose(compiled.negative_porosity, eager.negative_porosity)

    states = jax.tree.map(lambda value: jnp.stack((value, value)), state)
    mapped = jax.vmap(
        lambda candidate: (
            prepared.evaluate(candidate, parameters, 0.5).sei_current_density_a_m3
        )
    )(states)
    assert mapped.shape == (2, 4)

    def side_current_for_rate(rate):
        candidate = eqx.tree_at(
            lambda value: value.sei_reaction_rate_m_s,
            parameters,
            rate,
        )
        return prepared.evaluate(state, candidate, 0.5).side_current_a

    derivative = jax.grad(side_current_for_rate)(jnp.asarray(1.0e-14))
    assert bool(jnp.isfinite(derivative))
    assert float(derivative) > 0.0


def test_orchestration_closes_lithium_charge_product_film_and_porosity_ledgers():
    parameters = _parameters()
    adapter = BrosaPlanellaSpmeSeiAdapter(
        BrosaPlanellaSpmeSeiPlan(
            5,
            negative_electrolyte_cell_count=3,
            separator_electrolyte_cell_count=2,
            positive_electrolyte_cell_count=3,
            weak_side_reaction_threshold=0.2,
            small_overpotential_threshold_v=0.2,
        )
    )
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(0.5), RestStepPlan(0.25), CurrentStepPlan(0.25))
    )
    protocol_values = BatteryProtocolValues(protocol, jnp.asarray((0.2, -0.05)))
    support = SupportTuple(
        "battery.simulation",
        {
            "model": "spme-sei-brosa-planella-widanage",
            "control": "prescribed-current",
            "thermal": False,
            "plating": False,
        },
    )
    profile = CapabilityProfile(
        "battery.spme-sei-test-candidate",
        "phydrax-tests",
        "candidate",
        (support,),
        released=False,
    )
    save_times = jnp.asarray((0.0, 0.25, 0.5, 0.75, 1.0))
    experiment = BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(
            (
                "voltage_v",
                "current_a",
                "total_lithium_mol",
                "sei_lithium_mol",
            )
        ),
        BatteryDiffraxSolvePlan(
            stepsize_controller=dfx.StepTo(ts=save_times),
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-11,
        ),
        save_times,
        profile,
        support,
    ).prepare()
    result = experiment.run(
        parameters,
        BrosaPlanellaSpmeSeiInitialCondition(0.5, 0.5),
        protocol_values,
    )

    assert int(result.application_status) == int(BatteryRunStatus.SUCCESS)
    ledger = result.ledger
    assert bool(ledger.lithium_conserved)
    assert bool(ledger.electrolyte_lithium_conserved)
    assert bool(ledger.charge_conserved)
    assert bool(ledger.product_conserved)
    assert bool(ledger.film_conserved)
    assert bool(ledger.porosity_conserved)
    assert bool(ledger.interfaces_conserved)
    assert bool(ledger.current_split_conserved)
    assert bool(ledger.rest_current_cancellation_conserved)
    assert bool(ledger.domain_valid)
    assert bool(ledger.finite)
    assert bool(ledger.successful)
    np.testing.assert_allclose(ledger.integrated_terminal_charge_c, 0.0875)
    np.testing.assert_allclose(
        ledger.integrated_side_charge_c,
        FARADAY * (ledger.final_sei_lithium_mol - ledger.initial_sei_lithium_mol),
    )
    np.testing.assert_allclose(
        ledger.total_lithium_conservation_residual_mol, 0.0, atol=1.0e-10
    )


def test_state_is_strictly_isothermal_sei_only_without_thermal_or_plating_fields():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    assert adapter.model_id == "battery:spme:sei:brosa-planella-widanage:isothermal"
    assert set(state.__dataclass_fields__) == {
        "negative_amount_mol",
        "positive_amount_mol",
        "electrolyte_amount_mol",
        "negative_porosity",
    }
    forbidden = ("temperature", "heat", "plating", "crack", "active_material_loss")
    assert all(
        not any(token in name for token in forbidden)
        for name in state.__dataclass_fields__
    )
    malformed = BrosaPlanellaSpmeSeiState(
        state.negative_amount_mol,
        state.positive_amount_mol,
        state.electrolyte_amount_mol,
        jnp.ones((5,)),
    )
    with np.testing.assert_raises(ValueError):
        adapter.problem(prepared, malformed, _runtime(parameters, 0.0))

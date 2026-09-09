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
from phydrax.applications.battery._spm import (
    _stable_asinh_ratio,
    _transport,
    PrescribedCurrentSpmAdapter,
    PrescribedCurrentSpmPlan,
    SpmInitialCondition,
    SpmParameters,
    SpmState,
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


def _diffusivity_table(coordinate, nodes, values, *, electrode):
    coordinate_unit = "1" if coordinate == "stoichiometry" else "mol/m3"
    return TabulatedPropertyLaw(
        jnp.asarray(nodes),
        jnp.asarray(values),
        value_bounds=(0.0, jnp.inf),
        quantity=f"{electrode}-solid-diffusivity",
        coordinate=coordinate,
        value_unit="m2/s",
        coordinate_unit=coordinate_unit,
        source_id=f"test:{electrode}-solid-diffusivity:{coordinate}",
    )


def _parameters(
    *,
    maximum_current=20.0,
    positive_thickness=1.0e-4,
    limiting_electrode="balanced",
    negative_diffusivity=None,
    positive_diffusivity=None,
):
    return SpmParameters(
        electrode_area_m2=0.1,
        negative_electrode_thickness_m=1.0e-4,
        positive_electrode_thickness_m=positive_thickness,
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
        negative_solid_diffusivity=(
            _constant(
                2.0e-14,
                (250.0, 350.0),
                quantity="negative-solid-diffusivity",
                value_unit="m2/s",
            )
            if negative_diffusivity is None
            else negative_diffusivity
        ),
        positive_solid_diffusivity=(
            _constant(
                1.0e-14,
                (250.0, 350.0),
                quantity="positive-solid-diffusivity",
                value_unit="m2/s",
            )
            if positive_diffusivity is None
            else positive_diffusivity
        ),
        negative_exchange_current_density=_constant(
            5.0, (250.0, 350.0), quantity="negative-exchange-current", value_unit="A/m2"
        ),
        positive_exchange_current_density=_constant(
            4.0, (250.0, 350.0), quantity="positive-exchange-current", value_unit="A/m2"
        ),
        negative_open_circuit_potential=_constant(
            0.1, (0.0, 1.0), quantity="negative-ocp", value_unit="V"
        ),
        positive_open_circuit_potential=_constant(
            4.1, (0.0, 1.0), quantity="positive-ocp", value_unit="V"
        ),
        limiting_electrode=limiting_electrode,
    )


def _adapter(shell_count=5):
    adapter = PrescribedCurrentSpmAdapter(PrescribedCurrentSpmPlan(shell_count))
    return adapter, adapter.prepare()


def _runtime(parameters, current, *, duration=10.0):
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
        SpmInitialCondition(negative, positive),
    )


def _observable(adapter, output, name):
    return output.values[..., adapter.observable_names.index(name)]


def test_capacity_balancing_and_declared_limiting_electrode_are_validated():
    balanced = _parameters()
    adapter, prepared = _adapter()
    observed = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        _state(adapter, prepared, balanced),
        _runtime(balanced, 0.0),
    )
    negative_capacity = _observable(adapter, observed, "negative_theoretical_capacity_c")
    positive_capacity = _observable(adapter, observed, "positive_theoretical_capacity_c")
    np.testing.assert_allclose(negative_capacity, positive_capacity)
    np.testing.assert_allclose(
        _observable(adapter, observed, "usable_capacity_c"), negative_capacity
    )
    np.testing.assert_allclose(
        _observable(adapter, observed, "negative_capacity_headroom_c"), 0.0
    )
    np.testing.assert_allclose(
        _observable(adapter, observed, "positive_capacity_headroom_c"), 0.0
    )

    limiting = _parameters(
        positive_thickness=1.25e-4,
        limiting_electrode="negative",
    )
    limiting_state = _state(adapter, prepared, limiting)
    limiting_observed = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        limiting_state,
        _runtime(limiting, 0.0),
    )
    assert (
        float(_observable(adapter, limiting_observed, "positive_capacity_headroom_c"))
        > 0.0
    )
    with pytest.raises(Exception, match="neither balanced"):
        _parameters(positive_thickness=1.25e-4, limiting_electrode="balanced")
    with pytest.raises(Exception, match="declared limiting"):
        _parameters(positive_thickness=1.25e-4, limiting_electrode="positive")


def test_passive_positive_current_moves_lithium_negativeward_and_preserves_total():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    current = 8.0
    runtime = _runtime(parameters, current)
    problem = adapter.problem(prepared, state, runtime)
    rate = problem.drift(jnp.asarray(0.0), state, runtime)

    np.testing.assert_allclose(jnp.sum(rate.negative_amount_mol), current / FARADAY)
    np.testing.assert_allclose(jnp.sum(rate.positive_amount_mol), -current / FARADAY)
    np.testing.assert_allclose(
        jnp.sum(rate.negative_amount_mol) + jnp.sum(rate.positive_amount_mol),
        0.0,
        atol=1.0e-16,
    )


def test_current_to_outward_flux_uses_exact_representative_particle_area_factors():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    current = 6.0
    output = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        state,
        _runtime(parameters, current),
    )
    negative_solid_volume = (
        parameters.electrode_area_m2
        * parameters.negative_electrode_thickness_m
        * parameters.negative_active_material_volume_fraction
    )
    positive_solid_volume = (
        parameters.electrode_area_m2
        * parameters.positive_electrode_thickness_m
        * parameters.positive_active_material_volume_fraction
    )
    negative_area = 3.0 * negative_solid_volume / parameters.negative_particle_radius_m
    positive_area = 3.0 * positive_solid_volume / parameters.positive_particle_radius_m
    np.testing.assert_allclose(
        _observable(adapter, output, "negative_interfacial_flux_mol_m2_s"),
        -current / (FARADAY * negative_area),
    )
    np.testing.assert_allclose(
        _observable(adapter, output, "positive_interfacial_flux_mol_m2_s"),
        current / (FARADAY * positive_area),
    )
    np.testing.assert_allclose(
        _observable(adapter, output, "negative_specific_surface_area_m2_m3"),
        3.0
        * parameters.negative_active_material_volume_fraction
        / parameters.negative_particle_radius_m,
    )


def test_shell_diffusivity_laws_use_runtime_stoichiometry_or_concentration():
    stoichiometry_nodes = jnp.asarray((0.1, 0.5, 0.9))
    concentration_nodes = 3.0e4 * stoichiometry_nodes
    diffusivity_values = jnp.asarray((1.0e-14, 2.0e-14, 4.0e-14))
    stoichiometry_law = _diffusivity_table(
        "stoichiometry",
        stoichiometry_nodes,
        diffusivity_values,
        electrode="negative",
    )
    concentration_law = _diffusivity_table(
        "solid_lithium_concentration",
        concentration_nodes,
        diffusivity_values,
        electrode="negative",
    )
    target_stoichiometry = jnp.linspace(0.2, 0.8, 5)

    def evaluate(law):
        parameters = _parameters(negative_diffusivity=law)
        adapter, prepared = _adapter(5)
        uniform = _state(adapter, prepared, parameters)
        state = SpmState(
            uniform.negative_amount_mol * target_stoichiometry / 0.5,
            uniform.positive_amount_mol,
        )
        transport = _transport(
            prepared,
            state,
            parameters,
            jnp.asarray(0.0),
        )[1]
        return parameters, prepared, state, transport

    stoichiometry_parameters, prepared, state, stoichiometry_transport = evaluate(
        stoichiometry_law
    )
    _, _, _, concentration_transport = evaluate(concentration_law)
    expected = stoichiometry_law.evaluate(target_stoichiometry).values
    np.testing.assert_allclose(
        stoichiometry_transport.negative_diffusivity_m2_s,
        expected,
    )
    np.testing.assert_allclose(
        concentration_transport.negative_diffusivity_m2_s,
        expected,
    )
    np.testing.assert_allclose(
        stoichiometry_transport.negative_minimum_diffusivity_m2_s,
        jnp.min(expected),
    )
    np.testing.assert_allclose(
        stoichiometry_transport.positive_diffusivity_m2_s,
        jnp.full((5,), 1.0e-14),
    )
    assert bool(stoichiometry_transport.domain_valid)
    assert bool(concentration_transport.domain_valid)

    compiled = jax.jit(
        lambda candidate: (
            _transport(
                prepared,
                candidate,
                stoichiometry_parameters,
                jnp.asarray(0.0),
            )[1].negative_diffusivity_m2_s
        )
    )(state)
    batched_state = jax.tree.map(lambda value: jnp.stack((value, value)), state)
    mapped = jax.vmap(
        lambda candidate: (
            _transport(
                prepared,
                candidate,
                stoichiometry_parameters,
                jnp.asarray(0.0),
            )[1].negative_diffusivity_m2_s
        )
    )(batched_state)
    np.testing.assert_allclose(compiled, expected)
    np.testing.assert_allclose(mapped, jnp.stack((expected, expected)))


def test_shell_diffusivity_support_failure_invalidates_the_runtime_state():
    bounded_law = _diffusivity_table(
        "stoichiometry",
        (0.1, 0.9),
        (1.0e-14, 3.0e-14),
        electrode="negative",
    )
    parameters = _parameters(negative_diffusivity=bounded_law)
    adapter, prepared = _adapter(5)
    uniform = _state(adapter, prepared, parameters)
    unsupported_state = SpmState(
        uniform.negative_amount_mol.at[-1].multiply(1.9),
        uniform.positive_amount_mol,
    )
    transport = _transport(
        prepared,
        unsupported_state,
        parameters,
        jnp.asarray(0.0),
    )[1]
    output = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        unsupported_state,
        _runtime(parameters, 0.0),
    )

    assert not bool(transport.domain_valid)
    assert not bool(output.domain_valid)
    assert bool(jnp.all(jnp.isfinite(output.values)))


def test_soc_coordinates_agree_for_consistent_inventory_and_report_mismatch():
    parameters = _parameters()
    adapter, prepared = _adapter()
    consistent = _state(
        adapter,
        prepared,
        parameters,
        negative=0.3,
        positive=0.7,
    )
    consistent_output = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        consistent,
        _runtime(parameters, 0.0),
    )
    np.testing.assert_allclose(
        _observable(adapter, consistent_output, "negative_soc"), 0.25
    )
    np.testing.assert_allclose(
        _observable(adapter, consistent_output, "positive_soc"), 0.25
    )
    np.testing.assert_allclose(
        _observable(adapter, consistent_output, "soc_mismatch"),
        0.0,
        atol=2.0e-15,
    )

    inconsistent = _state(
        adapter,
        prepared,
        parameters,
        negative=0.5,
        positive=0.7,
    )
    inconsistent_output = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        inconsistent,
        _runtime(parameters, 0.0),
    )
    np.testing.assert_allclose(
        _observable(adapter, inconsistent_output, "negative_soc"), 0.5
    )
    np.testing.assert_allclose(
        _observable(adapter, inconsistent_output, "positive_soc"), 0.25
    )
    np.testing.assert_allclose(
        _observable(adapter, inconsistent_output, "soc_mismatch"), 0.25
    )


def test_symmetric_butler_volmer_voltage_has_passive_charge_and_discharge_signs():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)

    def voltage(current):
        output = adapter.observe(
            prepared,
            jnp.asarray(0.0),
            state,
            _runtime(parameters, current),
        )
        return _observable(adapter, output, "voltage_v")

    discharge_voltage = voltage(-8.0)
    rest_voltage = voltage(0.0)
    charge_voltage = voltage(8.0)
    assert float(discharge_voltage) < float(rest_voltage) < float(charge_voltage)
    np.testing.assert_allclose(rest_voltage, 4.0)

    charge_output = adapter.observe(
        prepared, jnp.asarray(0.0), state, _runtime(parameters, 8.0)
    )
    assert float(_observable(adapter, charge_output, "negative_overpotential_v")) < 0.0
    assert float(_observable(adapter, charge_output, "positive_overpotential_v")) > 0.0


def test_inverse_asinh_ratio_remains_finite_when_direct_division_overflows():
    numerator = jnp.asarray(1.0e30)
    denominator = jnp.asarray(1.0e-30)
    value = _stable_asinh_ratio(numerator, denominator)
    expected = jnp.log(jnp.asarray(2.0)) + jnp.log(numerator) - jnp.log(denominator)
    assert bool(jnp.isfinite(value))
    np.testing.assert_allclose(value, expected, rtol=2.0e-6)


def test_concentration_property_and_current_support_fail_closed_with_finite_outputs():
    parameters = _parameters(maximum_current=10.0)
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)

    unsupported_current = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        state,
        _runtime(parameters, 11.0),
    )
    assert not bool(unsupported_current.domain_valid)
    assert bool(jnp.all(jnp.isfinite(unsupported_current.values)))
    np.testing.assert_allclose(
        _observable(adapter, unsupported_current, "current_envelope_margin_a"), -1.0
    )

    negative_amount = state.negative_amount_mol.at[-1].set(-1.0)
    invalid_state = SpmState(negative_amount, state.positive_amount_mol)
    invalid_concentration = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        invalid_state,
        _runtime(parameters, 0.0),
    )
    assert not bool(invalid_concentration.domain_valid)
    assert bool(jnp.all(jnp.isfinite(invalid_concentration.values)))

    overfilled_state = SpmState(
        3.0 * state.negative_amount_mol,
        state.positive_amount_mol,
    )
    overfilled_concentration = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        overfilled_state,
        _runtime(parameters, 0.0),
    )
    assert not bool(overfilled_concentration.domain_valid)
    assert bool(jnp.all(jnp.isfinite(overfilled_concentration.values)))

    unsupported_temperature = eqx.tree_at(
        lambda profile: profile.temperature_k,
        parameters,
        jnp.asarray(400.0),
    )
    invalid_property = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        state,
        _runtime(unsupported_temperature, 0.0),
    )
    assert not bool(invalid_property.domain_valid)
    assert bool(jnp.all(jnp.isfinite(invalid_property.values)))


def test_tabulated_ocp_support_failure_is_a_domain_failure():
    parameters = _parameters()
    narrow_ocp = TabulatedPropertyLaw(
        jnp.asarray((0.2, 0.8)),
        jnp.asarray((0.1, 0.2)),
        quantity="negative-ocp",
        coordinate="stoichiometry",
        value_unit="V",
        coordinate_unit="1",
        source_id="test:narrow-negative-ocp",
    )
    unsupported = eqx.tree_at(
        lambda profile: profile.negative_open_circuit_potential,
        parameters,
        narrow_ocp,
    )
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters, negative=0.1, positive=0.9)
    output = adapter.observe(
        prepared,
        jnp.asarray(0.0),
        state,
        _runtime(unsupported, 0.0),
    )
    assert not bool(output.domain_valid)


def test_radial_refinement_reduces_boundary_reconstruction_distance():
    parameters = _parameters(
        positive_diffusivity=_diffusivity_table(
            "stoichiometry",
            (0.1, 0.5, 0.9),
            (0.5e-14, 1.0e-14, 2.0e-14),
            electrode="positive",
        )
    )
    coarse_adapter, coarse = _adapter(4)
    fine_adapter, fine = _adapter(8)
    coarse_state = _state(coarse_adapter, coarse, parameters)
    fine_state = _state(fine_adapter, fine, parameters)
    coarse_output = coarse_adapter.observe(
        coarse,
        jnp.asarray(0.0),
        coarse_state,
        _runtime(parameters, 8.0),
    )
    fine_output = fine_adapter.observe(
        fine,
        jnp.asarray(0.0),
        fine_state,
        _runtime(parameters, 8.0),
    )
    bulk = 0.5 * parameters.positive_maximum_concentration_mol_m3
    coarse_offset = bulk - _observable(
        coarse_adapter, coarse_output, "positive_surface_concentration_mol_m3"
    )
    fine_offset = bulk - _observable(
        fine_adapter, fine_output, "positive_surface_concentration_mol_m3"
    )
    np.testing.assert_allclose(fine_offset, 0.5 * coarse_offset, rtol=1.0e-5)


def test_spm_problem_and_observations_are_jittable_vmappable_and_fixed_path_differentiable():
    parameters = _parameters()
    adapter, prepared = _adapter()
    state = _state(adapter, prepared, parameters)
    runtime = _runtime(parameters, 4.0)
    problem = adapter.problem(prepared, state, runtime)

    eager_rate = problem.drift(jnp.asarray(0.0), state, runtime)
    compiled_rate = jax.jit(
        lambda time, candidate: problem.drift(time, candidate, runtime)
    )(jnp.asarray(0.0), state)
    np.testing.assert_allclose(
        compiled_rate.negative_amount_mol, eager_rate.negative_amount_mol
    )
    compiled_output = jax.jit(
        lambda time, candidate: adapter.observe(prepared, time, candidate, runtime)
    )(jnp.asarray(0.0), state)
    assert bool(compiled_output.domain_valid)

    states = jax.tree.map(lambda value: jnp.stack((value, value)), state)
    mapped = jax.vmap(
        lambda one_state: adapter.observe(
            prepared,
            jnp.asarray(0.0),
            one_state,
            runtime,
        )
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


def test_protocol_orchestration_conserves_lithium_charge_and_current():
    parameters = _parameters()
    adapter = PrescribedCurrentSpmAdapter(PrescribedCurrentSpmPlan(5))
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(2.0), RestStepPlan(1.0), CurrentStepPlan(2.0))
    )
    values = BatteryProtocolValues(protocol, jnp.asarray((5.0, -2.0)))
    support = SupportTuple(
        "battery.simulation",
        {"model": "spm", "control": "prescribed-current", "thermal": False},
    )
    profile = CapabilityProfile(
        "battery.spm-test-candidate",
        "phydrax-tests",
        "candidate",
        (support,),
        released=False,
    )
    save_times = jnp.asarray((0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    experiment = BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(("voltage_v", "current_a", "negative_soc", "positive_soc")),
        BatteryDiffraxSolvePlan(
            stepsize_controller=dfx.StepTo(ts=save_times),
            relative_tolerance=1.0e-7,
            absolute_tolerance=1.0e-9,
        ),
        save_times,
        profile,
        support,
    ).prepare()
    result = experiment.run(
        parameters,
        SpmInitialCondition(0.5, 0.5),
        values,
    )

    assert int(result.application_status) == int(BatteryRunStatus.SUCCESS)
    assert bool(result.ledger.lithium_conserved)
    assert bool(result.ledger.charge_conserved)
    assert bool(result.ledger.current_conserved)
    assert bool(result.ledger.successful)
    np.testing.assert_allclose(result.ledger.integrated_terminal_charge_c, 6.0)
    assert bool(jnp.all(result.outputs.valid))

    def final_negative_soc(first_current):
        dynamic_values = BatteryProtocolValues(
            protocol, jnp.stack((first_current, jnp.asarray(-2.0)))
        )
        dynamic_result = experiment.run(
            parameters,
            SpmInitialCondition(0.5, 0.5),
            dynamic_values,
        )
        return dynamic_result.outputs.values[-1, 2]

    fixed_path_derivative = jax.grad(final_negative_soc)(jnp.asarray(5.0))
    negative_capacity = (
        FARADAY
        * parameters.electrode_area_m2
        * parameters.negative_electrode_thickness_m
        * parameters.negative_active_material_volume_fraction
        * parameters.negative_maximum_concentration_mol_m3
        * (
            parameters.negative_stoichiometry_at_full
            - parameters.negative_stoichiometry_at_empty
        )
    )
    np.testing.assert_allclose(
        fixed_path_derivative,
        2.0 / negative_capacity,
        rtol=2.0e-5,
    )

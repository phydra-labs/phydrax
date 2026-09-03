#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.cardiovascular.circulation._devices import (
    Cannula,
    ControlEvent,
    ECMOCircuitPlan,
    evaluate_pump_map,
    HydraulicOxygenator,
    initialize_pacemaker_controller,
    initialize_pump_controller,
    PacemakerControllerPlan,
    PumpControllerPlan,
    PumpHeadFlowMap,
    PumpMapStatus,
    replay_pacemaker_controller,
    replay_pump_controller,
    run_ecmo_circuit,
    solve_ecmo_hydraulics,
    step_pacemaker_controller,
    step_pump_controller,
    TubingSegment,
)
from phydrax.applications.cardiovascular.circulation._oxygen import (
    BloodOxygenModel,
    evaluate_oxygen_content,
    exchange_membrane_oxygen,
    initialize_oxygen_transport_state,
    invert_oxygen_content,
    invert_oxygen_saturation,
    MembraneOxygenatorModel,
    mix_oxygen_content,
    OxygenTransportInputs,
    OxygenTransportPlan,
    step_oxygen_transport,
)
from phydrax.applications.cardiovascular.circulation._vascular_1d import (
    CharacteristicTerminal,
    couple_0d_pressure_port,
    initialize_vascular_state,
    reflect_characteristic_wave,
    solve_vascular_junction,
    SquareRootTubeLaw,
    step_vascular_1d,
    Vascular0DPort,
    Vascular1DPlan,
    VascularBoundaryState,
    VascularJunctionPlan,
    VascularStepStatus,
)


jax.config.update("jax_enable_x64", True)


def _pump_map() -> PumpHeadFlowMap:
    return PumpHeadFlowMap(
        "qualification-pump",
        [0.0, 2.0, 4.0],
        [2_000.0, 4_000.0],
        [[6.0, 4.0, 2.0], [12.0, 8.0, 4.0]],
    )


def _hydraulic_plan(*, oxygen_model=None) -> ECMOCircuitPlan:
    return ECMOCircuitPlan(
        _pump_map(),
        Cannula(
            "drainage",
            400.0,
            6.0,
            quadratic_loss_kPa_ms2_per_mm6=0.02,
        ),
        Cannula(
            "return",
            300.0,
            5.0,
            quadratic_loss_kPa_ms2_per_mm6=0.02,
        ),
        (
            TubingSegment(
                "circuit-tube",
                1_500.0,
                9.5,
                quadratic_loss_kPa_ms2_per_mm6=0.02,
            ),
        ),
        HydraulicOxygenator(
            "membrane-hydraulics",
            0.2,
            quadratic_loss_kPa_ms2_per_mm6=0.05,
        ),
        oxygen_model=oxygen_model,
    )


def test_tube_law_wave_speed_conservative_step_and_0d_port():
    law = SquareRootTubeLaw(100.0, 10.0, reference_pressure_kPa=8.0)
    expected_speed = np.sqrt(10.0 / (2.0 * 1.06))
    np.testing.assert_allclose(law.wave_speed(100.0, 1.06), expected_speed)

    runtime = Vascular1DPlan(
        "aorta",
        8,
        80.0,
        0.1,
        dynamic_viscosity_mg_per_mm_ms=0.0,
    ).prepare(law)
    area = jnp.asarray([100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0])
    state = initialize_vascular_state(runtime, area, jnp.zeros_like(area))
    result = step_vascular_1d(
        runtime,
        state,
        VascularBoundaryState(area[-1], state.flow_mm3_per_ms[-1]),
        VascularBoundaryState(area[0], state.flow_mm3_per_ms[0]),
    )
    assert bool(result.evidence.successful)
    np.testing.assert_allclose(
        result.evidence.mass_balance_residual_mm3, 0.0, atol=1.0e-11
    )
    np.testing.assert_allclose(
        result.evidence.momentum_balance_residual_mm4_per_ms, 0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        jnp.sum(result.state.area_mm2), jnp.sum(state.area_mm2), atol=1.0e-12
    )

    inlet = couple_0d_pressure_port(
        Vascular0DPort("aorta", "inlet"), law, 1.06, 100.0, 0.0, 9.0
    )
    outlet = couple_0d_pressure_port(
        Vascular0DPort("aorta", "outlet"), law, 1.06, 100.0, 0.0, 9.0
    )
    assert bool(inlet.successful) and bool(outlet.successful)
    assert float(inlet.flow_into_vessel_mm3_per_ms) > 0.0
    assert float(outlet.flow_into_vessel_mm3_per_ms) > 0.0


def test_vascular_step_rejects_boundary_driven_cfl_violation():
    law = SquareRootTubeLaw(100.0, 10.0, reference_pressure_kPa=8.0)
    runtime = Vascular1DPlan(
        "boundary-cfl",
        8,
        80.0,
        0.1,
        dynamic_viscosity_mg_per_mm_ms=0.0,
    ).prepare(law)
    area = jnp.full((8,), 100.0)
    state = initialize_vascular_state(runtime, area, jnp.zeros_like(area))
    result = step_vascular_1d(
        runtime,
        state,
        VascularBoundaryState(jnp.asarray(100.0), jnp.asarray(10_000.0)),
        VascularBoundaryState(jnp.asarray(100.0), jnp.asarray(0.0)),
    )
    assert not bool(result.evidence.successful)
    assert float(result.evidence.maximum_courant) > runtime.plan.maximum_courant
    assert int(result.evidence.status) & int(VascularStepStatus.CFL_VIOLATION)
    np.testing.assert_array_equal(result.state.area_mm2, state.area_mm2)
    np.testing.assert_array_equal(result.state.flow_mm3_per_ms, state.flow_mm3_per_ms)


def test_characteristic_reflection_and_junction_conservation():
    matched = CharacteristicTerminal("matched", 8.0, 0.5)
    reflection = reflect_characteristic_wave(matched, 1.2, 0.5)
    assert bool(reflection.successful)
    np.testing.assert_allclose(reflection.reflection_coefficient, 0.0)
    np.testing.assert_allclose(reflection.reflected_pressure_wave_kPa, 0.0)
    np.testing.assert_allclose(
        (reflection.terminal_pressure_kPa - 8.0) / reflection.terminal_flow_mm3_per_ms,
        0.5,
    )

    junction = solve_vascular_junction(
        VascularJunctionPlan("bifurcation", ("parent", "left", "right")),
        jnp.asarray([10.0, 8.0, 7.0]),
        jnp.asarray([0.4, 0.8, 1.2]),
    )
    assert bool(junction.successful)
    np.testing.assert_allclose(
        junction.conservation_residual_mm3_per_ms, 0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(junction.pressure_residual_kPa, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        jnp.sum(junction.branch_flow_away_mm3_per_ms), 0.0, atol=1.0e-14
    )


def test_pump_map_interpolates_and_refuses_extrapolation():
    pump_map = _pump_map()
    inside = evaluate_pump_map(pump_map, 1.0, 3_000.0)
    assert bool(inside.successful)
    np.testing.assert_allclose(inside.head_kPa, 7.5)

    outside = evaluate_pump_map(pump_map, 4.1, 3_000.0)
    assert not bool(outside.successful)
    assert np.isnan(float(outside.head_kPa))
    assert int(outside.status) & int(PumpMapStatus.FLOW_OUT_OF_DOMAIN)


def test_pacemaker_is_causal_rate_limited_and_exactly_replayable():
    plan = PacemakerControllerPlan(60.0, 120.0, 250.0, 2.0, 4.0)
    initial = initialize_pacemaker_controller(plan)
    times = jnp.asarray([250.0, 500.0, 750.0, 1_000.0, 1_250.0, 1_500.0])
    sensed = jnp.asarray([False, True, False, False, False, False])
    first = replay_pacemaker_controller(plan, initial, times, sensed)
    second = replay_pacemaker_controller(plan, initial, times, sensed)
    np.testing.assert_array_equal(first.pacing_output_mA, second.pacing_output_mA)
    np.testing.assert_array_equal(first.event, second.event)
    assert int(first.event[-1]) & int(ControlEvent.PACED)
    assert int(first.final_state.paced_count) == 1

    rejected = step_pacemaker_controller(plan, first.final_state, 1_400.0, False)
    assert not bool(rejected.successful)
    np.testing.assert_allclose(rejected.state.time_ms, first.final_state.time_ms)
    np.testing.assert_allclose(rejected.pacing_output_mA, 0.0)


def test_pump_controller_future_samples_cannot_change_prior_commands_and_replay():
    plan = PumpControllerPlan(
        10.0,
        300.0,
        2.0,
        3_000.0,
        2_000.0,
        4_000.0,
        20.0,
        minimum_integral_mm3=-100.0,
        maximum_integral_mm3=100.0,
    )
    initial = initialize_pump_controller(plan)
    times = jnp.asarray([10.0, 20.0, 30.0, 40.0])
    measured = jnp.asarray([2.0, 2.2, 2.4, 2.6])
    setpoint = jnp.full_like(measured, 3.0)
    first = replay_pump_controller(plan, initial, times, measured, setpoint)
    replayed = replay_pump_controller(plan, initial, times, measured, setpoint)
    changed_future = replay_pump_controller(
        plan, initial, times, measured.at[-1].set(0.0), setpoint
    )
    np.testing.assert_array_equal(first.speed_command_rpm, replayed.speed_command_rpm)
    np.testing.assert_array_equal(
        first.speed_command_rpm[:-1], changed_future.speed_command_rpm[:-1]
    )
    assert np.all(np.diff(np.asarray(first.speed_command_rpm)) <= 200.0)

    rejected = step_pump_controller(plan, first.final_state, 40.0, 2.5, 3.0)
    assert not bool(rejected.successful)
    np.testing.assert_allclose(rejected.state.speed_rpm, first.final_state.speed_rpm)


def test_hydraulic_ecmo_conserves_pressure_and_requires_explicit_oxygen_model():
    hydraulic_only = _hydraulic_plan()
    assert not hydraulic_only.gas_exchange_enabled
    assert not hydraulic_only.oxygenator.supports_gas_exchange
    hydraulics = solve_ecmo_hydraulics(hydraulic_only, 4_000.0, 1.0, 3.0)
    assert bool(hydraulics.successful)
    np.testing.assert_allclose(hydraulics.pressure_balance_residual_kPa, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(
        hydraulics.pump_head_kPa,
        hydraulics.pressure_load_kPa
        + hydraulics.drainage_drop_kPa
        + hydraulics.tubing_drop_kPa
        + hydraulics.oxygenator_drop_kPa
        + hydraulics.return_drop_kPa,
        atol=1.0e-8,
    )
    no_exchange = run_ecmo_circuit(hydraulic_only, 4_000.0, 1.0, 3.0, 12.0)
    assert bool(no_exchange.successful)
    assert not bool(no_exchange.gas_exchange_enabled)
    assert not bool(no_exchange.gas_exchange_performed)
    np.testing.assert_allclose(no_exchange.outlet_oxygen_content_mL_per_dL, 12.0)
    np.testing.assert_allclose(no_exchange.oxygen_transfer_mL_per_ms, 0.0)

    blood = BloodOxygenModel(14.0)
    membrane = MembraneOxygenatorModel(
        blood,
        20.0,
        1.0,
        minimum_flow_mm3_per_ms=0.1,
        maximum_flow_mm3_per_ms=4.0,
    )
    enabled = run_ecmo_circuit(
        _hydraulic_plan(oxygen_model=membrane), 4_000.0, 1.0, 3.0, 12.0
    )
    assert bool(enabled.successful)
    assert bool(enabled.gas_exchange_enabled)
    assert bool(enabled.gas_exchange_performed)
    assert float(enabled.outlet_oxygen_content_mL_per_dL) > 12.0
    assert float(enabled.oxygen_transfer_mL_per_ms) > 0.0


def test_oxygen_components_inversion_mixing_and_transport_are_conservative():
    model = BloodOxygenModel(15.0)
    content = evaluate_oxygen_content(model, 10.0)
    assert bool(content.successful)
    np.testing.assert_allclose(
        content.total_mL_per_dL,
        content.dissolved_mL_per_dL + content.bound_mL_per_dL,
    )
    saturation_inverse = invert_oxygen_saturation(model, content.saturation)
    total_inverse = invert_oxygen_content(model, content.total_mL_per_dL)
    assert bool(saturation_inverse.successful) and bool(total_inverse.successful)
    np.testing.assert_allclose(
        saturation_inverse.content.partial_pressure_kPa, 10.0, rtol=1.0e-11
    )
    np.testing.assert_allclose(
        total_inverse.content.partial_pressure_kPa, 10.0, rtol=1.0e-10
    )

    mixing = mix_oxygen_content(jnp.asarray([2.0, 3.0]), jnp.asarray([10.0, 20.0]))
    assert bool(mixing.successful)
    np.testing.assert_allclose(mixing.mixed_content_mL_per_dL, 16.0)
    np.testing.assert_allclose(mixing.conservation_residual_mL_per_ms, 0.0)

    transport = OxygenTransportPlan([100.0, 100.0], [0], [1], 1.0)
    state = initialize_oxygen_transport_state(transport, [10.0, 20.0])
    result = step_oxygen_transport(
        transport,
        state,
        OxygenTransportInputs(
            edge_flow_mm3_per_ms=jnp.asarray([5.0]),
            inflow_mm3_per_ms=jnp.empty((0,)),
            inflow_content_mL_per_dL=jnp.empty((0,)),
            outflow_mm3_per_ms=jnp.empty((0,)),
        ),
    )
    assert bool(result.evidence.successful)
    np.testing.assert_allclose(
        result.evidence.conservation_residual_mL, 0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        result.evidence.previous_inventory_mL,
        result.evidence.candidate_inventory_mL,
        atol=1.0e-15,
    )


def test_membrane_exchange_refuses_zero_flow_without_epsilon_division():
    model = BloodOxygenModel(14.0)
    membrane = MembraneOxygenatorModel(
        model,
        20.0,
        1.0,
        minimum_flow_mm3_per_ms=0.1,
        maximum_flow_mm3_per_ms=4.0,
    )
    refused = exchange_membrane_oxygen(membrane, 12.0, 0.0)
    assert not bool(refused.successful)
    assert np.isnan(float(refused.outlet.total_mL_per_dL))

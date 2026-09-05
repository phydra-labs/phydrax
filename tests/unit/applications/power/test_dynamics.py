import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.power._dynamics import (
    ClassicalMachine,
    DroopGovernor,
    FirstOrderAVR,
    initialize_power_dynamics,
    initialize_smib,
    Order4Machine,
    PowerEvent,
    simulate_power_dynamics,
)
from phydrax.applications.power._network import (
    Branch,
    Bus,
    BusControl,
    compile_network,
    Generator,
    Load,
    PowerNetwork,
    PowerStudy,
)
from phydrax.applications.power._power_flow import solve_power_flow
from phydrax.solver import DAEAdaptivePolicy, DAEConsistencyPolicy, DAESolvePolicy


def _network():
    return compile_network(
        PowerNetwork(
            buses=(Bus("grid"), Bus("machine")),
            branches=(
                Branch("line-a", "machine", "grid", r=0.01, x=0.4),
                Branch("line-b", "machine", "grid", r=0.01, x=0.4),
            ),
            generators=(Generator("unit", "machine", p=0.6),),
            loads=(Load("local", "machine", p=0.1, q=0.03),),
            base_mva=100.0,
        ),
        PowerStudy(
            (
                BusControl("grid", kind="reference"),
                BusControl("machine", kind="pv", voltage=1.02),
            )
        ),
    )


def _classical():
    return ClassicalMachine(
        "unit",
        inertia=3.5,
        damping=1.0,
        xd_prime=0.25,
        base_mva=50.0,
        stator_resistance=0.01,
    )


def _order4(generator="unit"):
    return Order4Machine(
        generator,
        inertia=3.5,
        xd=1.8,
        xq=1.6,
        xd_prime=0.3,
        xq_prime=0.55,
        td0_prime=5.0,
        tq0_prime=0.7,
        damping=1.0,
        base_mva=50.0,
        stator_resistance=0.01,
        avr=FirstOrderAVR(gain=20.0, time_constant=0.05, lower=-10, upper=10),
        governor=DroopGovernor(droop=0.05, time_constant=0.4, lower=-3, upper=3),
    )


@pytest.mark.parametrize("machine", (_classical(), _order4()))
def test_machine_equilibrium_preserves_pf_power_on_unequal_mva_bases(machine):
    compiled = _network()
    pf = solve_power_flow(compiled)
    initialized = initialize_smib(compiled, pf, machine, infinite_bus="grid")
    assert initialized.valid
    assert jnp.max(jnp.abs(initialized.operating_residual)) < 1e-7
    assert (
        jnp.max(
            jnp.abs(
                initialized.consistency.state_rate[: initialized.model.differential_size]
            )
        )
        < 1e-7
    )
    state = initialized.problem.initial_state
    assert jnp.allclose(
        initialized.model.machine_power(state), pf.generator_power, atol=1e-7
    )
    current = initialized.model.machine_currents(state)[0]
    terminal_voltage = initialized.model.voltage(state)[1]
    # Generated current is outward; machine current is twice network pu current
    # on this 50 MVA machine in the 100 MVA system. No extra factor of three.
    assert jnp.allclose(
        terminal_voltage * jnp.conj(current) * 0.5, pf.generator_power[0], atol=1e-7
    )
    result = simulate_power_dynamics(initialized, np.linspace(0.0, 0.02, 5))
    assert result.valid
    assert jnp.allclose(result.final_state, state, rtol=1e-6, atol=1e-7)


def test_multimachine_equilibrium_has_no_implicit_infinite_bus():
    compiled = compile_network(
        PowerNetwork(
            buses=(Bus("reference"), Bus("plant")),
            branches=(Branch("line", "reference", "plant", r=0.01, x=0.2),),
            generators=(
                Generator("reference-unit", "reference"),
                Generator("plant-a", "plant", p=0.2),
                Generator("plant-b", "plant", p=0.2),
            ),
            loads=(Load("demand", "plant", p=0.8, q=0.2),),
        ),
        PowerStudy(
            (
                BusControl("reference", kind="reference"),
                BusControl("plant", kind="pv", voltage=1.01),
            )
        ),
    )
    pf = solve_power_flow(compiled)
    initialized = initialize_power_dynamics(
        compiled,
        pf,
        (
            ClassicalMachine("reference-unit", inertia=4.0, xd_prime=0.3),
            ClassicalMachine("plant-a", inertia=3.0, xd_prime=0.25, base_mva=50),
            _order4("plant-b"),
        ),
    )
    assert initialized.valid
    assert initialized.equilibrium_norm < 1e-7
    assert jnp.allclose(
        initialized.model.machine_power(initialized.problem.initial_state),
        pf.generator_power,
        atol=1e-7,
    )
    # Multiple machines inject at plant, while the former PF reference is now
    # an ordinary machine terminal rather than an ideal voltage constraint.
    result = simulate_power_dynamics(initialized, np.linspace(0.0, 0.01, 3))
    assert result.valid
    assert jnp.allclose(result.final_state, initialized.problem.initial_state, atol=1e-7)


def test_fault_clear_and_breakers_reconstruct_voltage_without_state_jumps():
    compiled = _network()
    initialized = initialize_smib(
        compiled, solve_power_flow(compiled), _classical(), infinite_bus="grid"
    )
    result = simulate_power_dynamics(
        initialized,
        np.linspace(0.0, 0.1, 21),
        events=(
            PowerEvent(0.02, "fault", "machine", admittance=2.0 - 1.0j),
            PowerEvent(0.04, "clear", "machine"),
            PowerEvent(0.06, "trip", "line-a"),
            PowerEvent(0.08, "reclose", "line-a"),
        ),
    )
    assert result.valid
    model = initialized.model
    fault, clear, trip, reclose = result.events
    assert abs(model.voltage(fault.after)[1]) < abs(model.voltage(fault.before)[1])
    assert abs(model.voltage(clear.after)[1]) > abs(model.voltage(clear.before)[1])
    for index, event in enumerate(result.events):
        assert event.applied
        assert jnp.array_equal(
            event.differential_jump, jnp.zeros(model.differential_size)
        )
        assert jnp.max(jnp.abs(event.residual_after)) < 1e-7
        assert event.scheduled.event_count == 1
        assert jnp.allclose(event.scheduled.event_times[0], event.event.time, atol=1e-12)
        assert jnp.array_equal(result.segments[index].solution.states[-1], event.before)
        assert jnp.allclose(
            result.segments[index + 1].solution.states[0], event.after, atol=1e-8
        )
        assert result.segments[index + 1].solution.step_history.orders[0] == 1
    assert not trip.topology_after.branch_closed[0]
    assert reclose.topology_after.branch_closed[0]
    assert jnp.allclose(
        reclose.topology_after.admittance,
        initialized.model.initial_topology.admittance,
        atol=1e-12,
    )
    assert abs(result.final_state[0] - initialized.problem.initial_state[0]) > 1e-6


def test_inadmissible_restart_is_not_adopted_and_later_work_is_not_run():
    compiled = _network()
    initialized = initialize_smib(
        compiled, solve_power_flow(compiled), _classical(), infinite_bus="grid"
    )
    result = simulate_power_dynamics(
        initialized,
        np.linspace(0.0, 0.06, 13),
        events=(
            PowerEvent(0.02, "fault", "machine", admittance=2.0),
            PowerEvent(0.04, "clear", "machine"),
        ),
        consistency_policy=DAEConsistencyPolicy(0.0, 0.0, 0.0),
    )
    assert not result.valid
    assert result.status == "event_failed"
    failed, skipped = result.events
    assert not failed.applied
    assert not failed.consistency.admissible
    assert failed.consistency.state_correction_norm > 0
    assert skipped.status == "not_run"
    assert all(segment.status == "not_run" for segment in result.segments[1:])
    assert jnp.array_equal(result.final_state, failed.before)
    assert jnp.allclose(result.final_time, 0.02)


def test_source_free_island_failure_is_explicit():
    compiled = compile_network(
        PowerNetwork(
            buses=(Bus("grid"), Bus("machine"), Bus("load")),
            branches=(
                Branch("tie", "grid", "machine", r=0.01, x=0.2),
                Branch("feeder", "machine", "load", r=0.01, x=0.1),
            ),
            generators=(Generator("unit", "machine", p=0.6),),
            loads=(Load("demand", "load", p=0.2, q=0.05),),
        ),
        PowerStudy(
            (
                BusControl("grid", kind="reference"),
                BusControl("machine", kind="pv"),
                BusControl("load"),
            )
        ),
    )
    initialized = initialize_smib(
        compiled, solve_power_flow(compiled), _classical(), infinite_bus="grid"
    )
    result = simulate_power_dynamics(
        initialized,
        np.linspace(0.0, 0.04, 9),
        events=(PowerEvent(0.02, "trip", "feeder"),),
    )
    assert not result.valid
    assert result.events[0].status == "source_free_island"
    assert not result.events[0].applied
    assert result.segments[-1].status == "not_run"
    assert jnp.array_equal(result.final_state, result.events[0].before)


def test_unsupported_machine_coverage_and_controller_limits_fail_closed():
    compiled = _network()
    pf = solve_power_flow(compiled)
    with pytest.raises(ValueError, match="unknown generator"):
        initialize_smib(
            compiled, pf, ClassicalMachine("missing", inertia=3.0), infinite_bus="grid"
        )
    with pytest.raises(ValueError, match="governor command limits"):
        initialize_smib(
            compiled,
            pf,
            ClassicalMachine(
                "unit", inertia=3.0, governor=DroopGovernor(lower=0.0, upper=0.01)
            ),
            infinite_bus="grid",
        )
    with pytest.raises(ValueError, match="finite nonzero"):
        PowerEvent(0.0, "fault", "machine", admittance=complex(float("inf")))


def test_external_reference_is_explicit_and_pf_must_balance_this_network():
    compiled = _network()
    pf = solve_power_flow(compiled)
    with pytest.raises(ValueError, match="explicit infinite bus"):
        initialize_power_dynamics(compiled, pf, (_classical(),))
    changed = compile_network(
        PowerNetwork(
            buses=compiled.network.buses,
            branches=(
                Branch("line-a", "machine", "grid", r=0.01, x=0.8),
                compiled.network.branches[1],
            ),
            generators=compiled.network.generators,
            loads=compiled.network.loads,
        ),
        compiled.study,
    )
    with pytest.raises(ValueError, match="inconsistent with the supplied network"):
        initialize_smib(changed, pf, _classical(), infinite_bus="grid")


def test_load_fidelity_preserves_pq_by_default_and_impedance_only_when_requested():
    compiled = _network()
    pf = solve_power_flow(compiled)
    pq = initialize_smib(compiled, pf, _classical(), infinite_bus="grid")
    impedance = initialize_smib(
        compiled,
        pf,
        _classical(),
        infinite_bus="grid",
        load_model="constant_impedance",
    )
    depressed = 0.8 * pf.voltage
    pq_power = depressed * jnp.conj(pq.model.load_currents(depressed))
    impedance_power = depressed * jnp.conj(impedance.model.load_currents(depressed))
    assert jnp.allclose(pq_power, compiled.load_power, atol=1e-12)
    assert jnp.allclose(impedance_power, 0.8**2 * compiled.load_power, atol=1e-12)
    # A loaded zero voltage is genuinely singular for PQ, never regularized
    # to an invented impedance, while an impedance load has a defined zero.
    zero = jnp.zeros_like(pf.voltage)
    assert not jnp.isfinite(pq.model.load_currents(zero)[1])
    assert jnp.array_equal(impedance.model.load_currents(zero), zero)
    result = simulate_power_dynamics(impedance, np.linspace(0.0, 0.01, 3))
    assert result.valid
    assert result.load_model == "constant_impedance"


def test_fault_without_constant_power_solution_reports_native_restart_failure():
    compiled = _network()
    initialized = initialize_smib(
        compiled, solve_power_flow(compiled), _classical(), infinite_bus="grid"
    )
    # This shunt makes the Norton-source maximum transferable load power far
    # below the nonzero fixed P demand; impedance substitution would hide it.
    result = simulate_power_dynamics(
        initialized,
        np.linspace(0.0, 0.01, 3),
        events=(PowerEvent(0.0, "fault", "machine", admittance=1e4),),
    )
    assert not result.valid
    assert result.load_model == "constant_power"
    assert result.status == "event_failed"
    assert not result.events[0].consistency.initialization.valid
    assert not result.events[0].applied
    assert result.segments[0].status == "not_run"
    assert jnp.array_equal(result.final_state, initialized.problem.initial_state)


def test_off_grid_events_preserve_requested_samples_with_native_adaptive_default():
    compiled = _network()
    initialized = initialize_smib(
        compiled, solve_power_flow(compiled), _classical(), infinite_bus="grid"
    )
    requested = np.asarray((0.0, 0.02, 0.04))
    events = (
        PowerEvent(0.019, "fault", "machine", admittance=1.0),
        PowerEvent(0.025, "clear", "machine"),
    )
    result = simulate_power_dynamics(initialized, requested, events=events)
    assert result.valid
    samples = np.concatenate(
        [np.asarray(segment.solution.times) for segment in result.segments]
    )
    for time in (*requested, *(event.time for event in events)):
        assert time in samples
    assert all(event.applied for event in result.events)
    # A caller's fixed-grid policy is not silently replaced or relaxed.
    with pytest.raises(ValueError, match="max_step_ratio"):
        simulate_power_dynamics(
            initialized, requested, events=events, policy=DAESolvePolicy()
        )


def test_equilibrium_and_fault_satisfy_unchanged_native_constraint_certificate():
    compiled = compile_network(
        PowerNetwork(
            (Bus("grid", 110), Bus("machine", 110)),
            (Branch("tie", "grid", "machine", 0.0, 0.2),),
            (
                Generator("source", "grid", p=0.4),
                Generator("unit", "machine", p=0.6),
            ),
            (Load("demand", "grid", 1.0, 0.1),),
        ),
        PowerStudy((BusControl("grid", "reference"), BusControl("machine", "pv"))),
    )
    initialized = initialize_smib(
        compiled,
        solve_power_flow(compiled),
        ClassicalMachine("unit", inertia=4.0, damping=1.0, xd_prime=0.3),
        infinite_bus="grid",
    )
    requested = jnp.linspace(0.0, 0.2, 41)
    result = simulate_power_dynamics(
        initialized,
        requested,
        events=(
            PowerEvent(float(requested[10]), "fault", "machine", admittance=2 - 5j),
            PowerEvent(float(requested[20]), "clear", "machine"),
        ),
    )
    assert result.valid
    assert all(event.applied for event in result.events)
    native_acceptance = DAEAdaptivePolicy()
    algebraic = np.asarray(
        initialized.problem.system.structure.algebraic_equation_mask(
            initialized.problem.system.state_shape
        )
    )
    for segment in result.segments:
        solution = segment.solution
        assert jnp.all(solution.valid)
        assert jnp.all(solution.constraint_norm <= native_acceptance.constraint_tolerance)
        assert jnp.all(solution.residual_norm <= native_acceptance.residual_tolerance)
        problem = initialized.model.problem(solution.states[0], topology=segment.topology)
        for time, state, rate in zip(
            solution.times, solution.states, solution.state_rates, strict=True
        ):
            physical = np.asarray(problem.system.evaluate(time, state, rate))
            constraint_rms = np.sqrt(np.mean(np.square(physical[algebraic])))
            assert constraint_rms <= native_acceptance.constraint_tolerance

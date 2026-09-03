#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.applications.cardiovascular.circulation._closed_loop import (
    biventricular_closed_loop,
    replace_chamber_with_mechanics,
    systemic_closed_loop,
)
from phydrax.applications.cardiovascular.circulation._components import (
    Compliance,
    Inertance,
    MechanicsChamberCoupling,
    PressureSource,
    rc_pressure_transient,
    Resistance,
    StorageOwner,
    WindkesselRCR,
)
from phydrax.applications.cardiovascular.circulation._coronary import (
    coronary_closed_loop,
)
from phydrax.applications.cardiovascular.circulation._ledger import (
    audit_passivity,
    audit_pressure_volume_cycle,
    audit_total_volume,
    audit_valve_events,
    record_valve_event,
    ValveEventLedger,
)
from phydrax.applications.cardiovascular.circulation._network import (
    circulation_state_values,
    CirculationNetwork,
    initialize_consistent_state,
    prepare_consistent_initialization,
    PressureFlowConnection,
)
from phydrax.applications.cardiovascular.circulation._periodic import (
    commit_periodic_state,
    PeriodicShootingPlan,
    prepare_periodic_shooting,
    pressure_volume_work,
    solve_periodic_shooting,
)
from phydrax.applications.cardiovascular.circulation._valves import (
    ComplementarityValve,
    EventValve,
    SmoothValve,
)
from phydrax.dynamics import analyze_dae_structure, DAEStructuralPolicy


def test_rc_analytic_transient_and_passive_component_parameters():
    time = jnp.asarray([0.0, 2.0, 4.0])
    pressure = rc_pressure_transient(time, 0.0, 12.0, 2.0, 1.0)
    expected = 12.0 * (1.0 - jnp.exp(-time / 2.0))
    assert jnp.allclose(pressure, expected)

    resistance = Resistance("resistance", 2.0)
    compliance = Compliance(
        "compliance", 3.0, unstressed_volume=10.0, initial_volume=16.0
    )
    inertance = Inertance("inertance", 4.0)
    rcr = WindkesselRCR("rcr", 1.0, 3.0, 2.0, unstressed_volume=10.0, initial_volume=16.0)
    assert tuple(port.name for port in resistance.ports) == ("inlet", "outlet")
    assert compliance.storage_owner is StorageOwner.CIRCULATION
    assert jnp.isclose(compliance.stored_energy(16.0), 6.0)
    assert jnp.isclose(inertance.stored_energy(2.0), 8.0)
    assert jnp.isclose(rcr.stored_energy(16.0), 6.0)


def test_generic_dae_source_is_structurally_consistent_and_initializes():
    source = PressureSource("pump", 10.0)
    resistance = Resistance("load", 2.0)
    network = CirculationNetwork(
        (source, resistance),
        (
            PressureFlowConnection("pump", "outlet", "load", "inlet"),
            PressureFlowConnection("load", "outlet", "pump", "inlet"),
        ),
    )
    analysis = analyze_dae_structure(
        network.source, DAEStructuralPolicy(2, 16, tearing="automatic")
    )
    assert analysis.successful

    prepared = prepare_consistent_initialization(network)
    result = initialize_consistent_state(prepared)
    values = circulation_state_values(result)
    assert result.evidence.successful
    assert result.evidence.scaled_residual_norm < 1.0e-9
    assert jnp.isclose(values["load.flow_out"], 5.0, atol=1.0e-8)
    assert jnp.isclose(
        values["pump.pressure_out"] - values["pump.pressure_in"],
        10.0,
        atol=1.0e-8,
    )


def test_reference_closed_loops_are_closed_and_structurally_square():
    systemic = systemic_closed_loop()
    for model in (systemic, biventricular_closed_loop()):
        assert model.network.closed
        analysis = analyze_dae_structure(
            model.network.source,
            DAEStructuralPolicy(2, 128, tearing="automatic"),
        )
        assert analysis.successful
        assert len(analysis.variable_names) == len(analysis.equation_names)
        assert model.reference_total_volume > 0.0
    initialized = initialize_consistent_state(
        prepare_consistent_initialization(systemic.network)
    )
    assert initialized.evidence.successful
    assert initialized.evidence.scaled_residual_norm < 1.0e-9
    coronary = coronary_closed_loop()
    assert coronary.network.closed
    assert analyze_dae_structure(
        coronary.network.source,
        DAEStructuralPolicy(2, 128, tearing="automatic"),
    ).successful


def test_valve_routes_are_distinct_and_event_transitions_are_deterministic():
    smooth = SmoothValve("smooth", 0.01, 100.0, pressure_width=0.1)
    complementarity = ComplementarityValve("ideal", 0.01, smoothing=0.0)
    event = EventValve(
        "event",
        0.01,
        100.0,
        opening_pressure=0.1,
        closing_pressure=-0.1,
        minimum_dwell_time=2.0,
    )
    assert smooth.flow(1.0) > 0.0
    assert jnp.isclose(complementarity.complementarity_residual(1.0, 100.0), 0.0)

    opening = event.propose_event(0.0, 1.0)
    repeated = event.propose_event(0.0, 1.0)
    assert opening.event_required
    assert jnp.array_equal(opening.direction, repeated.direction)
    assert opening.source_state_id == event.state.state_id
    rejected = event.commit_event(opening, accept=False)
    assert rejected.state.state_id == event.state.state_id
    ledger = record_valve_event(ValveEventLedger(), opening)
    opened = event.commit_event(opening)
    with pytest.raises(ValueError, match="source state does not match"):
        opened.commit_event(opening)
    blocked = opened.propose_event(1.0, -1.0)
    assert not blocked.event_required
    closing = opened.propose_event(2.0, -1.0)
    assert closing.event_required
    ledger = record_valve_event(ledger, closing)
    evidence = audit_valve_events(ledger, minimum_dwell_time=2.0)
    assert evidence.deterministic


def test_volume_passivity_and_pressure_volume_work_ledgers():
    time = jnp.linspace(0.0, 1.0, 101)
    volume = jnp.full_like(time, 5_000.0)
    volume_ledger = audit_total_volume(time, volume, tolerance=1.0e-12)
    assert volume_ledger.conserved

    stored = jnp.exp(-2.0 * time)
    dissipated = 2.0 * jnp.exp(-2.0 * time)
    passivity = audit_passivity(
        time,
        jnp.zeros_like(time),
        stored,
        dissipated_power=dissipated,
        tolerance=1.0e-4,
    )
    assert passivity.passive

    pressure = jnp.asarray([1.0, 10.0, 10.0, 1.0, 1.0])
    chamber_volume = jnp.asarray([120.0, 120.0, 80.0, 80.0, 120.0])
    assert jnp.isclose(pressure_volume_work(pressure, chamber_volume), 360.0)
    work = audit_pressure_volume_cycle(pressure, chamber_volume)
    assert work.closed
    assert jnp.isclose(work.chamber_work, 360.0)


def test_periodic_shooting_closes_affine_cycle_map_and_commits():
    plan = PeriodicShootingPlan(800.0, (2,))
    prepared = prepare_periodic_shooting(
        plan,
        lambda state, cycle, args: 0.5 * state + jnp.asarray([1.0, -2.0]),
        "affine-contracting-cycle",
    )
    candidate = solve_periodic_shooting(prepared, jnp.zeros((2,)))
    committed = commit_periodic_state(candidate)
    assert candidate.evidence.successful
    assert candidate.evidence.maximum_absolute_closure < 1.0e-10
    assert jnp.allclose(committed.state, jnp.asarray([2.0, -4.0]))


def test_mechanics_replacement_transfers_storage_exclusively():
    model = systemic_closed_loop()
    coupling = MechanicsChamberCoupling(
        "left_ventricle",
        "mechanics-lv",
        lambda time, args: jnp.asarray(0.0),
    )
    replaced = replace_chamber_with_mechanics(
        model,
        "left_ventricle",
        coupling,
        135_000.0,
    )
    component = replaced.network.component("left_ventricle")
    assert component.storage_owner is StorageOwner.MECHANICS
    assert "left_ventricle.volume" not in replaced.network.storage_ids
    assert replaced.network.mechanics_storage_ids == ("left_ventricle",)
    assert replaced.external_storage[0].mechanics_chamber_id == "mechanics-lv"
    assert jnp.isclose(replaced.reference_total_volume, model.reference_total_volume)

    with pytest.raises(ValueError, match="preserve the chamber component name"):
        replace_chamber_with_mechanics(
            model,
            "left_ventricle",
            MechanicsChamberCoupling(
                "duplicate_left_ventricle",
                "mechanics-lv",
                lambda time, args: jnp.asarray(0.0),
            ),
            135_000.0,
        )

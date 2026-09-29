#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mixed FMU/native partitioned coupling through the explicit host boundary.

The FMU is the real compiled thermal-zone specimen
(`tests/interchange/data/thermal_zone.{c,xml}`): one lumped capacity
`C1 dT1/dt = G (T_b - T1) + P` with held inputs, integrated exactly per step.
The native participant is a node of capacity `C2`. Two physical couplings are
exercised:

- the FMU owns the conduction: the node temperature is held as the FMU boundary
  temperature, and the FMU's cumulative conducted heat is the node's whole-window
  heat loss;
- the node owns the conduction: the sampled FMU temperature drives the node's
  conduction, and the node's whole-window heat enters the FMU at a uniform rate.

Independent references: the closed-form discrete recursion of each coupling route
(exact per-window zone solution, constant-rate node update) and the analytic
two-capacity solution `T1 - T2 = (T1(0) - T2(0)) exp(-G (1/C1 + 1/C2) t)` with the
conserved heat content `C1 T1 + C2 T2`.
"""

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.interchange.fmi import (
    fmi_unit_definition,
    FMICoSimulationSession,
    FMICouplingBinding,
    FMICouplingParticipant,
    FMIVariableBinding,
    FMIWindowStatus,
    inspect_fmu,
)
from tests._support.fmi import (
    compile_specimen,
    description_only,
    package_fmu,
    SPECIMENS,
)


cpl = phx.solver.coupling
ZONE_CAPACITY = 2.0
CONDUCTANCE = 0.5
ZONE_START = 350.0
NODE_CAPACITY = 1.0
NODE_START = 300.0
FINAL_TIME = 1.0
LICENSE = "LicenseRef-PHYDRA-Proprietary"

SCALAR = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="lumped-scalar")
HEAT_FIELD = phx.discretization.DiscreteFieldSpace(
    "zone-node-heat",
    "zone-node-interface",
    phx.discretization.EntityDofLayout("zone-node-interface/entities", 1, 1),
    SCALAR,
    representation="cell_integral",
)
HEAT_MEASUREMENT = cpl.CouplingMeasurement.extensive(
    SCALAR, HEAT_FIELD.support_id, provenance_id="zone-node-conduction"
)
HEAT = cpl.CouplingQuantity("heat", phx.units.JOULE, sign_convention="node-to-zone")
TEMPERATURE = cpl.CouplingQuantity("temperature", phx.units.KELVIN)


def heat_port(port_id: str, direction: Any, /) -> Any:
    return cpl.CouplingPort(
        port_id,
        direction,
        SCALAR,
        field_space=HEAT_FIELD,
        quantity=HEAT,
        measurement=HEAT_MEASUREMENT,
        temporal_kind="interval_integral",
        reference_scale=1.0,
    )


def temperature_port(port_id: str, direction: Any, /) -> Any:
    return cpl.CouplingPort(
        port_id, direction, SCALAR, quantity=TEMPERATURE, reference_scale=1.0
    )


@pytest.fixture(scope="module")
def zone_fmus(tmp_path_factory: Any) -> dict[str, tuple[Path, str]]:
    """The compiled zone with and without advertised get/set FMU state."""
    root = tmp_path_factory.mktemp("thermal-zone-fmu").resolve()
    library = compile_specimen(root, "thermal_zone")
    description = (SPECIMENS / "thermal_zone.xml").read_text(encoding="utf-8")
    unrestorable = description.replace(
        'canGetAndSetFMUstate="true"', 'canGetAndSetFMUstate="false"'
    ).replace('canSerializeFMUstate="true"', 'canSerializeFMUstate="false"')
    return {
        "restore": package_fmu(root, library, description, "zone.fmu"),
        "none": package_fmu(root, library, unrestorable, "zone-no-restore.fmu"),
    }


@pytest.fixture
def zone_description(tmp_path: Any) -> Any:
    """Inspection-only archive: binding checks need no runtime or compiler."""
    root = tmp_path.resolve()
    path, digest = description_only(
        root,
        (SPECIMENS / "thermal_zone.xml").read_text(encoding="utf-8"),
        "zone.fmu",
    )
    return inspect_fmu(path.name, sha256=digest, trusted_root=root)


def open_zone(
    archive: tuple[Path, str],
    /,
    *,
    conductance: float = CONDUCTANCE,
    start: float = ZONE_START,
    maximum: float = 1e9,
) -> FMICoSimulationSession:
    path, digest = archive
    return FMICoSimulationSession(
        path.name,
        sha256=digest,
        trusted_root=path.parent,
        license_id=LICENSE,
        start_values={
            "capacity": ZONE_CAPACITY,
            "conductance": conductance,
            "initial_temperature": start,
            "maximum_temperature": maximum,
        },
    )


# Scenario A: the FMU owns the conduction; the native node receives its heat.


def conducting_zone(session: FMICoSimulationSession, /) -> FMICouplingParticipant:
    binding = FMICouplingBinding(
        session.model,
        (
            FMIVariableBinding(
                "boundary_temperature",
                temperature_port("zone/boundary-temperature", "input"),
                "hold",
            ),
            FMIVariableBinding(
                "conducted_heat", heat_port("zone/conducted-heat", "output"), "increment"
            ),
        ),
        time_unit=phx.units.SECOND,
    )
    return FMICouplingParticipant(session, binding, subsystem_id="zone")


def receiving_node() -> Any:
    """Node spending the received heat at a uniform rate; `args` scales it.

    `args` is 1 for the physical model; a non-finite value injects a native fault.
    """

    def rate(time: Any, state: Any, args: Any) -> Any:
        del time
        return jnp.full_like(state, args)

    def bind(window: Any, views: Any, model_state: Any, key: Any, args: Any) -> Any:
        del key
        return cpl.MethodWindowBinding(
            -args * views[0][0] / (window.size * NODE_CAPACITY), model_state
        )

    return cpl.FixedStepCouplingParticipant(
        phx.solver.SSPRK33FixedStepMethod(rate),
        bind,
        lambda native, args: (native,),
        subsystem_id="node",
        substeps=2,
        input_ports=(heat_port("node/heat", "input"),),
        output_ports=(temperature_port("node/temperature", "output"),),
    )


CONDUCTION_EXCHANGES = (
    cpl.CouplingExchange(
        "boundary-temperature", "node/temperature", "zone/boundary-temperature"
    ),
    cpl.CouplingExchange(
        "conducted-heat",
        "zone/conducted-heat",
        "node/heat",
        temporal=cpl.CouplingTemporalConversion("window-integral"),
    ),
)
ZONE_FIRST = cpl.CouplingSweep("gauss-seidel", subsystem_order=("zone", "node"))
EXPLICIT = cpl.ExplicitCouplingPolicy(ZONE_FIRST)
IMPLICIT = cpl.ImplicitCouplingPolicy(
    phx.nonlinear.FixedPointIteration(),
    phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-11,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=60,
    ),
    (
        cpl.CouplingTolerance("zone/boundary-temperature", absolute=1e-9),
        cpl.CouplingTolerance("node/heat", absolute=1e-9),
    ),
    fixed_point_sweep=ZONE_FIRST,
)


def conduction_plan(
    session: FMICoSimulationSession,
    policy: Any,
    window_size: float,
    /,
    *,
    differentiation: Any = None,
) -> Any:
    node = receiving_node()
    declaration = cpl.PartitionedCouplingDeclaration(
        (conducting_zone(session), node),
        CONDUCTION_EXCHANGES,
        policy,
        differentiation=differentiation,
        time_unit=phx.units.SECOND,
    )
    return cpl.prepare_host_coupling(
        declaration,
        {"node": node.initial_state(jnp.asarray([NODE_START]))},
        t0=0.0,
        window_size=window_size,
        args=jnp.asarray(1.0),
    )


def node_temperature(state: Any, /) -> float:
    native = state.participant_states[state.subsystem_ids.index("node")].native
    return float(np.asarray(native)[0])


def zone_temperature(session: FMICoSimulationSession, /) -> float:
    return session.get_values(("temperature",))["temperature"]


def discrete_conduction(window_size: float, windows: int, /, *, implicit: bool) -> Any:
    """Exact recursion of the zone-first coupling routes, window by window."""
    decay = math.exp(-CONDUCTANCE * window_size / ZONE_CAPACITY)
    ratio = ZONE_CAPACITY / NODE_CAPACITY * (1.0 - decay)
    zone, node = ZONE_START, NODE_START
    for _ in range(windows):
        # Explicit: the zone holds the accepted node temperature. Implicit: the
        # converged window holds the node temperature it produces.
        held = (node + ratio * zone) / (1.0 + ratio) if implicit else node
        heat = ZONE_CAPACITY * (1.0 - decay) * (held - zone)
        zone, node = held + (zone - held) * decay, node - heat / NODE_CAPACITY
    return zone, node


def analytic(time: float, /) -> tuple[float, float]:
    total = ZONE_CAPACITY + NODE_CAPACITY
    mean = (ZONE_CAPACITY * ZONE_START + NODE_CAPACITY * NODE_START) / total
    rate = CONDUCTANCE * (1.0 / ZONE_CAPACITY + 1.0 / NODE_CAPACITY)
    difference = (ZONE_START - NODE_START) * math.exp(-rate * time)
    return (
        mean + NODE_CAPACITY / total * difference,
        mean - ZONE_CAPACITY / total * difference,
    )


def heat_content(zone: float, node: float, /) -> float:
    return ZONE_CAPACITY * zone + NODE_CAPACITY * node


def observed_rates(errors: list[float], /) -> list[float]:
    return [math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:])]


# Binding: the model description decides type, causality, variability, and units.


def test_binding_resolves_exact_factors_and_restore_from_the_model_description(
    zone_description: Any,
) -> None:
    kilojoule = cpl.CouplingQuantity(
        "heat", phx.units.KILOJOULE, sign_convention="node-to-zone"
    )
    kilojoule_port = cpl.CouplingPort(
        "zone/conducted-heat",
        "output",
        SCALAR,
        field_space=HEAT_FIELD,
        quantity=kilojoule,
        measurement=HEAT_MEASUREMENT,
        temporal_kind="interval_integral",
        reference_scale=1.0,
    )
    binding = FMICouplingBinding(
        zone_description,
        (
            FMIVariableBinding(
                "heat_rate", heat_port("zone/heat", "input"), "uniform-rate"
            ),
            FMIVariableBinding("conducted_heat", kilojoule_port, "increment"),
            FMIVariableBinding(
                "temperature", temperature_port("zone/temperature", "output"), "sample"
            ),
        ),
        time_unit=phx.units.SECOND,
    )

    # J over s delivers W; kJ ports carry 1000 FMU joules per coordinate.
    assert binding.factors == (1.0, 1000.0, 1.0)
    assert binding.value_references == (1, 3, 2)
    assert binding.rollback == "restore"
    assert [unit.dimension for unit in binding.fmu_units] == [
        phx.units.ENERGY / phx.units.TIME,
        phx.units.ENERGY,
        phx.units.TEMPERATURE,
    ]
    assert fmi_unit_definition(zone_description.unit("J/K")).dimension == (
        phx.units.ENERGY / phx.units.TEMPERATURE
    )


@pytest.mark.parametrize(
    ("variable", "port", "realization", "match"),
    (
        (
            "temperature",
            temperature_port("zone/in", "input"),
            "hold",
            "causality 'output'",
        ),
        (
            "capacity",
            temperature_port("zone/in", "input"),
            "hold",
            "causality 'parameter'",
        ),
        (
            "temperature_celsius",
            temperature_port("zone/out", "output"),
            "sample",
            "affine",
        ),
        ("steps", temperature_port("zone/out", "output"), "sample", "Integer"),
        (
            "heat_rate",
            temperature_port("zone/in", "input"),
            "hold",
            "does not realize port",
        ),
        (
            "heat_rate",
            heat_port("zone/in", "input"),
            "uniform-rate",
            "not the coupling time unit",
        ),
        ("pressure", temperature_port("zone/in", "input"), "hold", "no variable"),
    ),
    ids=(
        "output-as-input",
        "parameter",
        "affine-unit",
        "integer",
        "dimension",
        "time-unit",
        "unknown-variable",
    ),
)
def test_binding_refuses_variables_the_model_description_does_not_support(
    zone_description: Any, variable: str, port: Any, realization: Any, match: str
) -> None:
    time_unit = (
        phx.units.MILLISECOND
        if match == "not the coupling time unit"
        else (phx.units.SECOND)
    )
    with pytest.raises(ValueError, match=match):
        FMICouplingBinding(
            zone_description,
            (FMIVariableBinding(variable, port, realization),),
            time_unit=time_unit,
        )


def test_variable_binding_refuses_ports_it_cannot_realize() -> None:
    with pytest.raises(ValueError, match="binds instantaneous output ports"):
        FMIVariableBinding("temperature", temperature_port("zone/in", "input"), "sample")
    untyped = cpl.CouplingPort("zone/out", "output", SCALAR, reference_scale=1.0)
    with pytest.raises(ValueError, match="physical quantity"):
        FMIVariableBinding("temperature", untyped, "sample")


# Execution: explicit, implicit, rollback, and the independent references.


def test_explicit_route_without_restore_conserves_heat_and_converges(
    zone_fmus: Any,
) -> None:
    initial = heat_content(ZONE_START, NODE_START)
    errors = []
    for window_size in (0.1, 0.05, 0.025):
        with open_zone(zone_fmus["none"]) as session:
            prepared = conduction_plan(session, EXPLICIT, window_size)
            solution = cpl.solve_host_coupling(
                prepared, t1=FINAL_TIME, args=jnp.asarray(1.0)
            )
            zone = zone_temperature(session)
            conducted = session.get_values(("conducted_heat",))["conducted_heat"]
        node = node_temperature(solution.final_state)
        windows = round(FINAL_TIME / window_size)

        assert solution.successful and len(solution.windows) == windows
        assert {result.commit for result in solution.windows} == {"accepted"}
        assert all(result.host_evaluations == 1 for result in solution.windows)
        assert all(result.restores == 0 for result in solution.windows)
        reference = discrete_conduction(window_size, windows, implicit=False)
        np.testing.assert_allclose((zone, node), reference, rtol=1e-12)
        assert abs(heat_content(zone, node) - initial) <= 1e-12 * initial
        ledger = np.asarray(solution.final_state.cumulative_exchange_budget)
        row = solution.final_state.budget_row_ids.index("conducted-heat")
        # The ledger credits exactly the heat the FMU reports having conducted.
        assert ledger[row, 0] + ledger[row, 1] == pytest.approx(0.0, abs=1e-12)
        assert ledger[row, 1] == pytest.approx(conducted, rel=1e-12)
        errors.append(
            max(abs(zone - analytic(FINAL_TIME)[0]), abs(node - analytic(FINAL_TIME)[1]))
        )

    assert all(0.9 < rate < 1.1 for rate in observed_rates(errors))


def test_implicit_route_replays_every_iterate_from_the_restored_fmu_state(
    zone_fmus: Any,
) -> None:
    initial = heat_content(ZONE_START, NODE_START)
    errors = []
    for window_size in (0.1, 0.05, 0.025):
        with open_zone(zone_fmus["restore"]) as session:
            prepared = conduction_plan(session, IMPLICIT, window_size)
            solution = cpl.solve_host_coupling(
                prepared, t1=FINAL_TIME, args=jnp.asarray(1.0)
            )
            zone = zone_temperature(session)
        node = node_temperature(solution.final_state)
        windows = round(FINAL_TIME / window_size)

        assert solution.successful and len(solution.windows) == windows
        for result in solution.windows:
            diagnostics = result.window.diagnostics
            assert result.commit == "accepted"
            assert bool(result.window.converged)
            # Every re-execution after the first starts from the restored checkpoint.
            assert result.host_evaluations > 1
            assert result.restores == result.host_evaluations - 1
            assert np.all(
                np.asarray(diagnostics.participant_evaluations) == result.host_evaluations
            )
            assert bool(diagnostics.counts_complete)
        reference = discrete_conduction(window_size, windows, implicit=True)
        np.testing.assert_allclose((zone, node), reference, rtol=0.0, atol=1e-8)
        assert abs(heat_content(zone, node) - initial) <= 1e-12 * initial
        errors.append(
            max(abs(zone - analytic(FINAL_TIME)[0]), abs(node - analytic(FINAL_TIME)[1]))
        )

    assert all(0.9 < rate < 1.1 for rate in observed_rates(errors))


def test_rejected_window_restores_the_fmu_and_the_retry_replays_bitwise(
    zone_fmus: Any,
) -> None:
    with open_zone(zone_fmus["restore"]) as session:
        prepared = conduction_plan(session, EXPLICIT, 0.1)
        first = cpl.advance_host_coupling_window(
            prepared, prepared.initial_state, jnp.asarray(1.0)
        ).accepted_state
        checkpoint = (zone_temperature(session), session.time)
        faulted = cpl.advance_host_coupling_window(prepared, first, jnp.asarray(jnp.nan))

        assert faulted.commit == "rejected" and faulted.restores == 1
        assert not bool(faulted.window.successful)
        # The native step owner rejects its non-finite step; the FMU step itself was OK.
        assert int(faulted.window.status) == int(cpl.CouplingStatus.PARTICIPANT_FAILURE)
        statuses = np.asarray(faulted.window.diagnostics.participant_statuses)
        zone_index = faulted.accepted_state.subsystem_ids.index("zone")
        assert int(statuses[zone_index]) == int(FMIWindowStatus.OK)
        # The FMU had completed its step; the rejection returned it to the checkpoint.
        assert (zone_temperature(session), session.time) == checkpoint
        assert node_temperature(faulted.accepted_state) == node_temperature(first)
        retried = cpl.advance_host_coupling_window(
            prepared, faulted.accepted_state, jnp.asarray(1.0)
        )
        retried_values = (
            zone_temperature(session),
            node_temperature(retried.accepted_state),
        )

    with open_zone(zone_fmus["restore"]) as session:
        prepared = conduction_plan(session, EXPLICIT, 0.1)
        undisturbed = cpl.solve_host_coupling(prepared, t1=0.2, args=jnp.asarray(1.0))
        expected = (zone_temperature(session), node_temperature(undisturbed.final_state))

    assert retried.commit == "accepted"
    assert retried_values == expected


def test_rejected_window_without_restore_is_unrecoverable_and_refuses_to_continue(
    zone_fmus: Any,
) -> None:
    with open_zone(zone_fmus["none"]) as session:
        prepared = conduction_plan(session, EXPLICIT, 0.1)
        start = prepared.initial_state
        faulted = cpl.advance_host_coupling_window(prepared, start, jnp.asarray(jnp.nan))

        assert faulted.commit == "unrecoverable" and faulted.restores == 0
        assert faulted.accepted_state.time == start.time
        assert session.time == pytest.approx(0.1)
        with pytest.raises(ValueError, match="cannot return to an accepted checkpoint"):
            cpl.advance_host_coupling_window(prepared, start, jnp.asarray(1.0))


def test_fmu_discard_rejects_the_window_and_restores_the_zone(zone_fmus: Any) -> None:
    # A zone colder than the node warms toward it; its cutout lies inside window one.
    with open_zone(zone_fmus["restore"], start=290.0, maximum=290.1) as session:
        prepared = conduction_plan(session, EXPLICIT, 0.1)
        result = cpl.advance_host_coupling_window(
            prepared, prepared.initial_state, jnp.asarray(1.0)
        )
        diagnostics = result.window.diagnostics
        zone_index = result.accepted_state.subsystem_ids.index("zone")

        assert result.commit == "rejected" and result.restores == 1
        assert int(result.window.status) == int(cpl.CouplingStatus.PARTICIPANT_FAILURE)
        assert int(np.asarray(diagnostics.participant_statuses)[zone_index]) == int(
            FMIWindowStatus.EARLY_RETURN
        )
        # The discarded partial step was undone by the FMU's native state restore.
        assert session.time == 0.0
        assert zone_temperature(session) == 290.0
        assert node_temperature(result.accepted_state) == NODE_START


# Scenario B: the native node owns the conduction; the FMU receives its heat.


def conducting_node() -> Any:
    def rate(time: Any, state: Any, zone: Any) -> Any:
        del time
        flow = CONDUCTANCE * (state[0] - zone)
        return jnp.stack((-flow / NODE_CAPACITY, flow))

    def bind(window: Any, views: Any, model_state: Any, key: Any, args: Any) -> Any:
        del window, key, args
        return cpl.MethodWindowBinding(views[0][0], model_state)

    return cpl.FixedStepCouplingParticipant(
        phx.solver.SSPRK33FixedStepMethod(rate),
        bind,
        lambda native, args: (),
        subsystem_id="node",
        substeps=4,
        input_ports=(temperature_port("node/zone-temperature", "input"),),
        output_ports=(heat_port("node/heat", "output"),),
        # The node's accumulator already spent the window's heat.
        amounts=lambda start, end, args: (end[1:] - start[1:],),
    )


def test_uniform_rate_and_sample_realize_native_conduction(zone_fmus: Any) -> None:
    initial = heat_content(ZONE_START, NODE_START)
    errors = []
    for window_size in (0.1, 0.05, 0.025):
        with open_zone(zone_fmus["none"], conductance=0.0) as session:
            zone_participant = FMICouplingParticipant(
                session,
                FMICouplingBinding(
                    session.model,
                    (
                        FMIVariableBinding(
                            "heat_rate", heat_port("zone/heat", "input"), "uniform-rate"
                        ),
                        FMIVariableBinding(
                            "temperature",
                            temperature_port("zone/temperature", "output"),
                            "sample",
                        ),
                    ),
                    time_unit=phx.units.SECOND,
                ),
                subsystem_id="zone",
            )
            node = conducting_node()
            declaration = cpl.PartitionedCouplingDeclaration(
                (zone_participant, node),
                (
                    cpl.CouplingExchange(
                        "zone-temperature", "zone/temperature", "node/zone-temperature"
                    ),
                    cpl.CouplingExchange(
                        "node-heat",
                        "node/heat",
                        "zone/heat",
                        temporal=cpl.CouplingTemporalConversion("window-integral"),
                    ),
                ),
                # The zone consumes the heat the node spent in the same sweep.
                cpl.ExplicitCouplingPolicy(
                    cpl.CouplingSweep("gauss-seidel", subsystem_order=("node", "zone"))
                ),
                time_unit=phx.units.SECOND,
            )
            prepared = cpl.prepare_host_coupling(
                declaration,
                {"node": node.initial_state(jnp.asarray([NODE_START, 0.0]))},
                t0=0.0,
                window_size=window_size,
            )
            solution = cpl.solve_host_coupling(prepared, t1=FINAL_TIME)
            zone = zone_temperature(session)
        node_value = node_temperature(solution.final_state)

        assert solution.successful
        assert all(
            int(
                np.asarray(result.window.diagnostics.participant_statuses)[
                    result.window.accepted_state.subsystem_ids.index("zone")
                ]
            )
            == int(FMIWindowStatus.OK)
            for result in solution.windows
        )
        assert abs(heat_content(zone, node_value) - initial) <= 1e-12 * initial
        ledger = np.asarray(solution.final_state.cumulative_exchange_budget)
        row = solution.final_state.budget_row_ids.index("node-heat")
        # The zone gained exactly the heat the ledger credits it.
        assert ledger[row, 1] == pytest.approx(
            ZONE_CAPACITY * (zone - ZONE_START), rel=1e-12
        )
        errors.append(
            max(
                abs(zone - analytic(FINAL_TIME)[0]),
                abs(node_value - analytic(FINAL_TIME)[1]),
            )
        )

    assert all(0.9 < rate < 1.1 for rate in observed_rates(errors))


# Refusals: native routes, retries without restore, and derivatives.


def test_native_preparation_refuses_host_participants_and_leaves_the_fmu_usable(
    zone_fmus: Any,
) -> None:
    with open_zone(zone_fmus["restore"]) as session:
        zone = conducting_zone(session)
        node = receiving_node()
        graph = cpl.CouplingGraph((zone, node), CONDUCTION_EXCHANGES)
        states = (
            cpl.HostParticipantState(jnp.asarray(0.0), 0),
            node.initial_state(jnp.asarray([NODE_START])),
        )
        values = (jnp.asarray([NODE_START]), jnp.asarray([0.0]))
        with pytest.raises(ValueError, match="JIT-capable.*prepare_host_coupling"):
            cpl.prepare_coupling(graph, states, values, policy=EXPLICIT)
        declaration = cpl.PartitionedCouplingDeclaration(
            (zone, node), CONDUCTION_EXCHANGES, EXPLICIT
        )
        with pytest.raises(ValueError, match="JIT-capable.*prepare_host_coupling"):
            cpl.lower_partitioned_coupling(
                declaration,
                {"zone": states[0], "node": states[1]},
                t0=0.0,
                t1=1.0,
                window_size=0.1,
            )
        prepared = conduction_plan(session, EXPLICIT, 0.1)
        with pytest.raises(ValueError, match="advance_host_coupling_window"):
            cpl.advance_coupling_window(prepared.plan, prepared.initial_state, 0.1)
        with pytest.raises(ValueError, match="prepare_host_coupling"):
            cpl.refresh_coupling(prepared.plan, graph)

        assert session.time == 0.0
        result = cpl.advance_host_coupling_window(
            prepared, prepared.initial_state, jnp.asarray(1.0)
        )
        assert result.commit == "accepted"


def test_implicit_route_requires_real_fmu_state_restore(zone_fmus: Any) -> None:
    with open_zone(zone_fmus["none"]) as session:
        zone = conducting_zone(session)
        assert zone.rollback == "none"
        with pytest.raises(ValueError, match="explicit non-retrying route"):
            conduction_plan(session, IMPLICIT, 0.1)
        # The refused preparation neither captured nor advanced the FMU.
        assert session.time == 0.0
        prepared = conduction_plan(session, EXPLICIT, 0.1)
        result = cpl.advance_host_coupling_window(
            prepared, prepared.initial_state, jnp.asarray(1.0)
        )
        assert result.commit == "accepted"


def test_derivative_requests_through_the_fmu_are_refused(zone_fmus: Any) -> None:
    with open_zone(zone_fmus["restore"]) as session:
        support = conducting_zone(session).derivative_support
        assert support.route == "none" and support.alternatives
        with pytest.raises(ValueError, match="no adjoint"):
            conduction_plan(
                session,
                EXPLICIT,
                0.1,
                differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
            )
        prepared = conduction_plan(session, EXPLICIT, 0.1)

        def zone_heat(scale: Any) -> Any:
            result = cpl.advance_host_coupling_window(
                prepared, prepared.initial_state, scale
            )
            return result.accepted_state.exchange_values[0][0]

        with pytest.raises(TypeError, match="JAX transformations"):
            jax.grad(zone_heat)(jnp.asarray(1.0))
        assert session.time == 0.0


def test_host_route_requires_a_host_participant() -> None:
    source = receiving_node()
    sink = cpl.FixedStepCouplingParticipant(
        phx.solver.SSPRK33FixedStepMethod(
            lambda time, state, args: jnp.zeros_like(state)
        ),
        lambda window, views, model_state, key, args: cpl.MethodWindowBinding(
            None, model_state
        ),
        lambda native, args: (),
        subsystem_id="sink",
        substeps=1,
        input_ports=(temperature_port("sink/temperature", "input"),),
    )
    declaration = cpl.PartitionedCouplingDeclaration(
        (source, sink),
        (cpl.CouplingExchange("probe", "node/temperature", "sink/temperature"),),
        EXPLICIT,
    )
    with pytest.raises(ValueError, match="all-native graph runs on the native runtime"):
        cpl.prepare_host_coupling(
            declaration,
            {
                "node": source.initial_state(jnp.asarray([NODE_START])),
                "sink": sink.initial_state(jnp.asarray([0.0])),
            },
            t0=0.0,
            window_size=0.1,
        )


def _named_zone(session: FMICoSimulationSession, name: str, /) -> Any:
    binding = FMICouplingBinding(
        session.model,
        (
            FMIVariableBinding(
                "boundary_temperature",
                temperature_port(f"{name}/boundary-temperature", "input"),
                "hold",
            ),
            FMIVariableBinding(
                "conducted_heat",
                heat_port(f"{name}/conducted-heat", "output"),
                "increment",
            ),
        ),
        time_unit=phx.units.SECOND,
    )
    return FMICouplingParticipant(session, binding, subsystem_id=name)


def _two_zone_plan(first: Any, second: Any, /) -> Any:
    """Zones `first` then `second` hold the node temperature; `second` heats the node."""
    node = receiving_node()
    names = (first.subsystem_id, second.subsystem_id)
    declaration = cpl.PartitionedCouplingDeclaration(
        (first, second, node),
        (
            *(
                cpl.CouplingExchange(
                    f"{name}-boundary", "node/temperature", f"{name}/boundary-temperature"
                )
                for name in names
            ),
            cpl.CouplingExchange(
                "heat",
                f"{names[1]}/conducted-heat",
                "node/heat",
                temporal=cpl.CouplingTemporalConversion("window-integral"),
            ),
        ),
        cpl.ExplicitCouplingPolicy(
            cpl.CouplingSweep("gauss-seidel", subsystem_order=(*names, "node"))
        ),
        time_unit=phx.units.SECOND,
    )
    return cpl.prepare_host_coupling(
        declaration,
        {"node": node.initial_state(jnp.asarray([NODE_START]))},
        t0=0.0,
        window_size=0.1,
        args=jnp.asarray(1.0),
    )


def _session(archive: tuple[Path, str], /, *, stop_time: float | None) -> Any:
    path, digest = archive
    return FMICoSimulationSession(
        path.name,
        sha256=digest,
        trusted_root=path.parent,
        license_id=LICENSE,
        stop_time=stop_time,
        start_values={
            "capacity": ZONE_CAPACITY,
            "conductance": CONDUCTANCE,
            "initial_temperature": ZONE_START,
            "maximum_temperature": 1e9,
        },
    )


def test_failed_host_call_restores_the_other_participants_and_keeps_its_error(
    zone_fmus: Any,
) -> None:
    # `first` advances window two; `second` then fails past its stop time and dies.
    with (
        _session(zone_fmus["restore"], stop_time=None) as first_session,
        _session(zone_fmus["restore"], stop_time=0.15) as second_session,
    ):
        prepared = _two_zone_plan(
            _named_zone(first_session, "first"), _named_zone(second_session, "second")
        )
        accepted = cpl.advance_host_coupling_window(
            prepared, prepared.initial_state, jnp.asarray(1.0)
        ).accepted_state

        with pytest.raises(Exception) as failure:
            cpl.advance_host_coupling_window(prepared, accepted, jnp.asarray(1.0))

        assert "closed" not in str(failure.value)
        assert any("'second'" in note for note in failure.value.__notes__)
        assert second_session.closed
        # The live participant returned to the accepted communication point.
        assert first_session.time == pytest.approx(float(accepted.time))


def test_rejected_window_restores_restorable_participants_beside_unrestorable_ones(
    zone_fmus: Any,
) -> None:
    with (
        _session(zone_fmus["restore"], stop_time=None) as restorable,
        _session(zone_fmus["none"], stop_time=None) as unrestorable,
    ):
        prepared = _two_zone_plan(
            _named_zone(restorable, "restorable"), _named_zone(unrestorable, "fixed")
        )
        start = prepared.initial_state
        temperature = restorable.get_values(("temperature",))["temperature"]

        faulted = cpl.advance_host_coupling_window(prepared, start, jnp.asarray(jnp.nan))

        assert faulted.commit == "unrecoverable" and faulted.restores == 1
        assert unrestorable.time == pytest.approx(0.1)
        assert restorable.time == 0.0
        assert restorable.get_values(("temperature",))["temperature"] == temperature


def test_host_windows_end_exactly_on_the_prepared_grid_and_at_t1(zone_fmus: Any) -> None:
    # 0.1 + 0.1 + 0.1 > 0.3: an accumulated clock steps past the FMU stop time.
    with _session(zone_fmus["restore"], stop_time=0.3) as session:
        prepared = conduction_plan(session, EXPLICIT, 0.1)

        solution = cpl.solve_host_coupling(prepared, t1=0.3, args=jnp.asarray(1.0))

        assert solution.successful
        assert [float(w.accepted_state.time) for w in solution.windows] == [
            0.1,
            0.2,
            0.3,
        ]
        assert session.time == 0.3


def test_host_route_runs_on_the_declared_coupling_clock(zone_fmus: Any) -> None:
    with open_zone(zone_fmus["restore"]) as session:
        node = receiving_node()
        states = {"node": node.initial_state(jnp.asarray([NODE_START]))}
        for time_unit, match in (
            (None, "declare the coupling time_unit"),
            (phx.units.MILLISECOND, "time unit other than the coupling clock"),
        ):
            declaration = cpl.PartitionedCouplingDeclaration(
                (conducting_zone(session), node),
                CONDUCTION_EXCHANGES,
                EXPLICIT,
                time_unit=time_unit,
            )
            with pytest.raises(ValueError, match=match):
                cpl.prepare_host_coupling(declaration, states, t0=0.0, window_size=0.1)
        assert session.time == 0.0


@pytest.mark.parametrize(
    ("variant", "time_unit", "match"),
    (
        ("implicit-seconds", phx.units.MILLISECOND, "time is in seconds"),
        ("independent-without-unit", phx.units.SECOND, "declares no unit"),
    ),
    ids=("implicit-seconds", "independent-without-unit"),
)
def test_binding_verifies_the_time_unit_of_the_fmu_independent_variable(
    tmp_path: Any, variant: str, time_unit: Any, match: str
) -> None:
    text = (SPECIMENS / "thermal_zone.xml").read_text(encoding="utf-8")
    declared = (
        '<ScalarVariable name="time" valueReference="8" causality="independent" '
        'variability="continuous"><Real unit="s"/></ScalarVariable>'
    )
    assert declared in text
    xml = {
        "implicit-seconds": text.replace(declared, ""),
        "independent-without-unit": text.replace(
            '<Real unit="s"/></Scalar', "<Real/></Scalar"
        ),
    }[variant]
    root = tmp_path.resolve()
    path, digest = description_only(root, xml, f"{variant}.fmu")
    model = inspect_fmu(path.name, sha256=digest, trusted_root=root)
    heat_rate = FMIVariableBinding(
        "heat_rate", heat_port("zone/heat", "input"), "uniform-rate"
    )

    with pytest.raises(ValueError, match=match):
        FMICouplingBinding(model, (heat_rate,), time_unit=time_unit)
    if variant == "implicit-seconds":
        binding = FMICouplingBinding(model, (heat_rate,), time_unit=phx.units.SECOND)
        assert binding.factors == (1.0,)

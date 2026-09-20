import jax.numpy as jnp

import phydrax as phx


def test_linear_gcode_and_process_graph():
    program = phx.manufacturing.parse_linear_gcode("G1 X10 F600\nG4 P1000\nG1 Y10")
    assert len(program.events) == 3
    graph = phx.manufacturing.ManufacturingProcessGraph.create(
        (
            phx.manufacturing.ManufacturingStage("deposit"),
            phx.manufacturing.ManufacturingStage("heat", ("deposit",)),
        )
    )
    assert len(graph.stages) == 2


def test_goldak_and_activation_are_executable():
    source = phx.manufacturing.GoldakDoubleEllipsoidSource(
        1000.0, 0.8, 0.01, 0.02, 0.005, 0.004
    )
    assert source.evaluate(jnp.zeros((1, 3)), jnp.zeros(3))[0] > 0
    state = phx.manufacturing.MaterialActivationState(
        jnp.asarray((True, False)), jnp.asarray((0.0, -1.0))
    ).activate(jnp.asarray((False, True)), 1.0)
    assert jnp.all(state.active)


def test_runtime_and_process_transfer_close_mass_and_energy_balances():
    event = phx.manufacturing.ToolpathEvent(
        "deposit",
        "deposit",
        0.0,
        2.0,
        "machine",
        (0.0,),
        (1.0,),
        power_w=100.0,
        mass_rate_kg_s=2.0,
    )
    runtime = phx.manufacturing.ManufacturingRuntime.create(
        phx.manufacturing.ProcessSchedule.create((event,)),
        jnp.asarray(((0.0,), (1.0,))),
        jnp.asarray((1.0, 1.0)),
        0.5,
    )
    result = runtime.advance(
        phx.manufacturing.ManufacturingRuntimeState.initialize(2), 2.0
    )
    assert jnp.isclose(jnp.sum(result.state.deposited_mass_kg), 4.0)
    assert jnp.isclose(jnp.sum(result.state.supplied_energy_j), 200.0)
    assert jnp.isclose(result.mass_balance_residual_kg, 0.0)
    assert jnp.isclose(result.energy_balance_residual_j, 0.0)

    transfer = phx.manufacturing.ProcessTransferPlan.create(
        jnp.asarray(((0.75, 0.25), (0.25, 0.75))),
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1.0, 1.0)),
    )
    target = transfer.transfer_runtime_state(result.state)
    assert jnp.isclose(
        jnp.sum(target.deposited_mass_kg),
        jnp.sum(result.state.deposited_mass_kg),
    )
    assert jnp.isclose(
        jnp.sum(target.supplied_energy_j),
        jnp.sum(result.state.supplied_energy_j),
    )

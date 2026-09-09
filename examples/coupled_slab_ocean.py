"""Run a nonmatching-grid, real-freshwater slab/hydrostatic-ocean exchange.

Run from the worktree: python examples/coupled_slab_ocean.py
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.geophysics._coupling import (
    coupling_surface_field,
    HydrostaticOceanCouplingSubsystem,
    SlabReservoir,
)
from phydrax.applications.ocean._hydrostatic_step import HydrostaticContinuationState
from phydrax.solver._partitioned_coupling_graph import CouplingGraph, prepare_coupling
from phydrax.solver._partitioned_coupling_runtime import advance_coupling_window
from phydrax.solver._partitioned_coupling_types import (
    CouplingExchange,
    CouplingSweep,
    CouplingTransferRequirement,
    ExplicitCouplingPolicy,
)


def slab_ocean_scenario(*, water_rate=1.0e-4, conductance=20.0, water_mass=10.0):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(2, periodic=axis < 2)
            for axis in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -10.0), (2.0, 2.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    geometry = phx.discretization.TensorZHydrostaticGridPlan(
        discretization,
        jnp.full((2, 2), 10.0),
        vertical_coordinate="zstar",
    ).prepare()
    ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(geometry).prepare()
    receiver = HydrostaticOceanCouplingSubsystem(ocean)
    field, measure = coupling_surface_field(jnp.asarray([4.0]), "slab-surface")
    slab = SlabReservoir(
        field,
        measure,
        dry_heat_capacity=5.0e5,
        conductance=conductance,
        water_rate=water_rate,
    )
    properties = phx.discretization.TransferProperties(
        conservative=True,
        constant_preserving=True,
        positivity_preserving=True,
    )

    def transfer(source, target, matrix):
        return phx.discretization.FieldTransfer(
            source,
            target,
            phx.linalg.DenseLinearOperator(
                matrix, source=source.vector_space, target=target.vector_space
            ),
            properties=properties,
        )

    forward = transfer(field, receiver.field, jnp.ones((4, 1)))
    backward = transfer(receiver.field, field, jnp.full((1, 4), 0.25))
    requirement = CouplingTransferRequirement(
        conservative=True,
        constant_preserving=True,
        positivity_preserving=True,
        frame_action="preserve",
    )
    exchanges = (
        CouplingExchange(
            "heat",
            slab.output_ports[0].port_id,
            receiver.input_ports[0].port_id,
            transfer=forward,
            requirement=requirement,
        ),
        CouplingExchange(
            "water",
            slab.output_ports[1].port_id,
            receiver.input_ports[1].port_id,
            transfer=forward,
            requirement=requirement,
        ),
        CouplingExchange(
            "temperature",
            receiver.output_ports[0].port_id,
            slab.input_ports[0].port_id,
            transfer=backward,
            requirement=requirement,
        ),
    )
    initial = HydrostaticContinuationState.initialize(
        ocean, ocean.initialize_state(jnp.zeros((2, 2)))
    )
    graph = CouplingGraph((slab, receiver), exchanges)
    prepared = prepare_coupling(
        graph,
        (slab.initialize(288.15, water_mass), initial),
        (jnp.zeros(4), jnp.zeros(4), jnp.full((1,), 283.15)),
        policy=ExplicitCouplingPolicy(
            CouplingSweep(
                "gauss-seidel",
                subsystem_order=(slab.subsystem_id, receiver.subsystem_id),
            )
        ),
    )
    return prepared, slab, receiver


def run():
    prepared, slab, receiver = slab_ocean_scenario()
    state = prepared.reference_state
    for _ in range(4):
        result = advance_coupling_window(prepared, state, 0.25)
        if not bool(result.successful):
            raise RuntimeError(
                f"Coupled window rejected with status {int(result.status)}"
            )
        state = result.accepted_state
    np.testing.assert_allclose(
        np.sum(state.cumulative_exchange_budget, axis=1), 0.0, atol=1e-10
    )
    slab_index = state.subsystem_ids.index(slab.subsystem_id)
    ocean_index = state.subsystem_ids.index(receiver.subsystem_id)
    initial_slab = prepared.reference_state.participant_states[slab_index]
    final_slab = state.participant_states[slab_index]
    continuation = state.participant_states[ocean_index]
    heat = (
        receiver.method.ocean.plan.reference_density
        * receiver.heat_capacity
        * continuation.ledger.tracer_change["conservative_temperature"]
    )
    water = receiver.freshwater_density * continuation.ledger.freshwater_volume
    np.testing.assert_allclose(
        slab.measure.integrate(final_slab.enthalpy - initial_slab.enthalpy) + heat,
        0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        slab.measure.integrate(final_slab.water_mass - initial_slab.water_mass) + water,
        0,
        atol=1e-12,
    )
    print(
        {
            "time_seconds": float(state.time),
            "heat_to_ocean_J": float(heat),
            "freshwater_to_ocean_kg": float(water),
            "accepted_exchange_budget": np.asarray(
                state.cumulative_exchange_budget
            ).tolist(),
        }
    )


if __name__ == "__main__":
    run()

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=1000.0,
        gas_density=10.0,
        surface_tension=0.072,
    )
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization, material
    ).prepare()
    x = (jnp.arange(8) + 0.5) / 8
    y = (jnp.arange(8) + 0.5) / 8
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    alpha = jnp.where((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.2**2, 1.0, 0.0)
    state = two_phase.initial_state(alpha)
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(state)
    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.001),
        None,
    )
    view = phx.applications.two_phase_flow.two_phase_diagnostic_view(
        two_phase, result.accepted_state
    )
    return {
        "successful": bool(result.successful),
        "liquid_volume": float(view.liquid_volume),
        "interface_measure": float(view.interface_measure),
        "topology_events": int(view.topology_event_count),
    }


if __name__ == "__main__":
    print(run())

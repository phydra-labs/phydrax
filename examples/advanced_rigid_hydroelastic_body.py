#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (4.0, 4.0, 0.0))))
    reference = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrodynamics",)
    ).prepare()
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        reference, jnp.full((4, 4), -1.0)
    )
    hydro = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface, coupling_iterations=4, coupling_tolerance=1.0e-7
    ).prepare()
    fluid = phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(
        hydro.initial_state(jnp.zeros((4, 4)))
    )
    markers = jnp.asarray(
        (
            (-0.1, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.0, -0.1, 0.0),
            (0.0, 0.1, 0.0),
            (0.0, 0.0, -0.1),
            (0.0, 0.0, 0.1),
        )
    )
    normals = markers / jnp.linalg.norm(markers, axis=-1)[:, None]
    body_plan = phx.applications.hydrodynamics.MappedRigidHydroelasticBodyPlan(
        markers,
        normals,
        jnp.ones((6,)),
        modal_basis=jnp.linspace(-1.0, 1.0, 6)[:, None],
        modal_mass=(1.0,),
        modal_stiffness=(2.0,),
        modal_damping=(0.1,),
        tolerance=1.0e-6,
    )
    body = body_plan.initial_state(position=(2.0, 2.0, -0.5))
    state = phx.applications.hydrodynamics.RigidHydroelasticContinuationState(
        fluid, body, jnp.asarray(0.0), jnp.asarray(0.0)
    )
    method = phx.applications.hydrodynamics.RigidHydroelasticALEMethod(
        phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(hydro),
        body_plan,
    )
    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(0.001),
        None,
    )
    return {
        "successful": bool(result.successful),
        "body_work": float(result.accepted_state.body_work),
        "modal_norm": float(
            jnp.linalg.norm(result.accepted_state.body.modal_coordinates)
        ),
    }


if __name__ == "__main__":
    print(run())

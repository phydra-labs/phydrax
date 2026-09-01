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
            phx.discretization.UniformCellAxisSpec(4, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (1.0, 1.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    ocean = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        phx.applications.ocean.LinearSeawaterReference(),
        coriolis_parameter=0.5,
    ).prepare(discretization)
    velocity = (
        jnp.ones(discretization.face_layouts[0].shape),
        jnp.zeros(discretization.face_layouts[1].shape),
        jnp.zeros(discretization.face_layouts[2].shape),
    )
    temperature = jnp.full(discretization.cell_shape, 10.0)
    salinity = jnp.full(discretization.cell_shape, 35.0)
    coordinates = ocean.initial_state(velocity, temperature, salinity)
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        coordinates
    )
    problem = phx.solver.FixedStepProblem(
        phx.applications.ocean.OceanBoussinesqSSPRK33Method(ocean),
        continuation,
        t0=0.0,
        t1=0.1,
        step_size=0.01,
        state_geometry=phx.metrix.EuclideanStateGeometry(),
    )
    solution = phx.solver.solve_fixed_step(problem)
    final = jax_tree_last(solution.states)
    view = ocean.state_view(final.coordinates)
    return {
        "successful": bool(solution.successful),
        "mean_u": float(jnp.mean(view.velocity[0])),
        "mean_v": float(jnp.mean(view.velocity[1])),
        "coriolis_work": float(final.coriolis_work),
    }


def jax_tree_last(tree):
    import jax

    return jax.tree.map(lambda leaf: leaf[-1], tree)


if __name__ == "__main__":
    print(run())

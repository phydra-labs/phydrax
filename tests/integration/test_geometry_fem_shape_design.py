#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _problem_components():
    coordinates = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.0, 0.0],
        ]
    )
    triangles = jnp.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=jnp.int32,
    )
    mesh = phx.discretization.CellMesh.from_triangles(coordinates, triangles)
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 1),
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    geometry = phx.geometry.Circle(
        (0.0, 0.0),
        1.0,
        feature_id="circle",
    ).compile()
    projection = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        coordinates[:4],
        0.3,
        source_id="circle-boundary",
    )
    motion = phx.discretization.FiniteElementMeshMotionPlan(
        discretization,
        projection,
    )
    return geometry, discretization, motion


def test_state_design_acceptance_uses_the_same_dynamic_fe_realization():
    geometry, discretization, motion = _problem_components()
    radius_index = geometry.schema.index(phx.geometry.ParameterId("circle", "radius"))
    design = geometry.state.replace_at(radius_index, jnp.asarray(1.1))

    def area(current_design):
        realization = motion.realize(current_design)
        blocks = discretization.evaluate_geometry(
            "u",
            realization.runtime.coordinates,
        )
        return sum(jnp.sum(block.physical_weights) for block in blocks)

    problem = phx.optim.StateDesignProblem(
        lambda state, current_design, _args: state - area(current_design),
        lambda state, _design, _args: 0.5 * state**2,
        state_admissibility=lambda _state, current_design, _args: (
            motion.realize(current_design).accepted
        ),
        state_realization=lambda _state, current_design, _args: (
            motion.realize(current_design).accepted
        ),
        problem_id="geometry-fe-shape-design",
    )

    result = problem.solve_state(design, jnp.asarray(0.0))

    assert bool(result.acceptance.accepted)
    assert jnp.allclose(result.state, jnp.asarray(2.0 * 1.1**2), atol=1.0e-7)
    assert jnp.linalg.norm(result.residual) < 1.0e-7

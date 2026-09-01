#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent synthetic qualification for barrier derivatives and conservative CCD."""

import json

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def main():
    distance_squared = jnp.asarray(0.04)
    activation = jnp.asarray(0.5)
    analytic = jax.grad(
        lambda value: phx.applications.contact.physical_clamped_log_barrier(
            value, activation, 0.0
        )
    )(distance_squared)
    step = 1.0e-6
    finite_difference = (
        phx.applications.contact.physical_clamped_log_barrier(
            distance_squared + step, activation, 0.0
        )
        - phx.applications.contact.physical_clamped_log_barrier(
            distance_squared - step, activation, 0.0
        )
    ) / (2.0 * step)

    source = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    moving_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0, 1)),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),)),
    )
    moving = phx.discretization.PreparedCollisionSurface(
        moving_plan,
        jnp.asarray(((-0.5, 0.5), (0.5, 0.5))),
        phx.discretization.selection_collision_operator(source, jnp.asarray((0, 1))),
    )
    static_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((10, 11)),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),)),
        pair_policy=phx.discretization.ContactPairPolicy(
            2, static_mask=jnp.ones((2,), dtype=bool)
        ),
    )
    static = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0))),
        phx.discretization.static_collision_operator(source, 2, 2),
    )
    scene = phx.discretization.PreparedCollisionScene((moving, static))
    start = scene.positions(source.zeros())
    end = scene.positions(jnp.broadcast_to(jnp.asarray((0.0, -1.0)), source.shape))
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    epoch = search.build(scene, start, end_positions=end)
    safety = phx.discretization.collision_free_step_limit(
        phx.discretization.InclusionCCDPlan(time_tolerance=1.0e-8),
        scene,
        epoch,
        start,
        end,
    )
    derivative_defect = float(jnp.abs(analytic - finite_difference))
    print(
        json.dumps(
            {
                "qualification": "deformable-contact",
                "device": str(jax.devices()[0]),
                "dtype": str(distance_squared.dtype),
                "barrier_derivative_defect": derivative_defect,
                "candidate_complete": bool(epoch.successful),
                "ccd_successful": bool(safety.successful),
                "ccd_step_size": float(safety.step_size),
                "ccd_interval_count": int(safety.interval_count),
                "qualified": bool(
                    derivative_defect < 1.0e-6
                    and epoch.successful
                    and safety.successful
                    and 0.0 < safety.step_size < 0.5
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-topology geometry realization feeding dynamic finite elements."""

import json

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _diamond_discretization():
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
    return phx.discretization.FiniteElementPlan(mesh, field).prepare()


def run():
    geometry = phx.geometry.Circle(
        (0.0, 0.0),
        1.0,
        feature_id="design-circle",
    ).compile()
    discretization = _diamond_discretization()
    projection = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        discretization.mesh.coordinates[:4],
        0.3,
        source_id="design-circle-boundary",
    )
    motion = phx.discretization.FiniteElementMeshMotionPlan(
        discretization,
        projection,
    )
    radius_index = geometry.schema.index(
        phx.geometry.ParameterId("design-circle", "radius")
    )

    def physical_area(radius):
        design = geometry.state.replace_at(radius_index, radius)
        realization = motion.realize(design)
        blocks = discretization.evaluate_geometry(
            "u",
            realization.runtime.coordinates,
        )
        return sum(jnp.sum(block.physical_weights) for block in blocks)

    design = geometry.state.replace_at(radius_index, jnp.asarray(1.1))
    realization = eqx.filter_jit(motion.realize)(design)
    area = physical_area(jnp.asarray(1.1))
    area_derivative = jax.grad(physical_area)(jnp.asarray(1.1))
    expired = motion.realize(geometry.state.replace_at(radius_index, jnp.asarray(1.8)))

    sphere = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id="display-sphere",
    ).compile()
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(7) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]]))
    surface_plan = phx.geometry.discover_implicit_surface(
        sphere,
        grid,
        policy=phx.geometry.ImplicitSurfacePolicy(maximum_intersection_pairs=500_000),
        source_id="display-sphere-surface",
    )
    surface = surface_plan.realize(sphere.state)

    return {
        "accepted": bool(realization.accepted),
        "topology_id": realization.evidence.topology_id,
        "geometry_layout_id": realization.evidence.geometry_layout_id,
        "area": float(area),
        "area_derivative": float(area_derivative),
        "minimum_relative_jacobian": float(
            realization.evidence.geometry.minimum_relative_jacobian
        ),
        "invalid_trial_accepted": bool(expired.accepted),
        "invalid_trial_refresh_required": bool(expired.refresh_required),
        "surface_accepted": bool(surface.accepted),
        "surface_topology_id": surface.evidence.topology_id,
        "surface_vertices": int(surface.vertices.shape[0]),
        "surface_faces": int(surface.faces.shape[0]),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))

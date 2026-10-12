# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Two shared-template grains with physical-area, symmetric level-set contact."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def main() -> None:
    depth = 4
    resolution = 2**depth
    coordinates = np.stack(
        np.meshgrid(*([np.arange(resolution)] * 3), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    grid = phx.discretization.SparseVoxelGridPlan(
        phx.discretization.MortonAddressPlan((-1.0,) * 3, (1.0,) * 3, depth),
        brick_size=2,
        brick_capacity=(resolution // 2) ** 3,
    ).prepare(coordinates)
    sphere = phx.geometry.Sphere((0.0,) * 3, 0.5, feature_id="grain").compile()
    enclosure = phx.geometry.ExactSDFEnclosureCertificate(sphere.field_certificate)
    samples = phx.geometry.VoxelGeometrySamplingPlan(
        grid, enclosure=enclosure, enclosure_geometry=sphere
    ).sample_level_set(sphere)
    mu, weight = np.polynomial.legendre.leggauss(32)
    phi = 2 * np.pi * np.arange(8) / 8
    nodes = 0.5 * np.stack(
        (
            np.broadcast_to(mu[:, None], (32, 8)),
            np.sqrt(1 - mu[:, None] ** 2) * np.cos(phi)[None, :],
            np.sqrt(1 - mu[:, None] ** 2) * np.sin(phi)[None, :],
        ),
        axis=-1,
    ).reshape(-1, 3)
    areas = np.broadcast_to((0.25 * weight * 2 * np.pi / 8)[:, None], (32, 8)).reshape(-1)
    template = phx.discretization.LevelSetShapeTemplatePlan(
        samples,
        np.arange(nodes.shape[0], dtype=np.int64),
        nodes,
        areas,
        0,
        exact_volume=4 * np.pi * 0.5**3 / 3,
        exact_center_of_mass=np.zeros(3),
        exact_inertia_per_mass=0.1 * np.eye(3),
    )
    shapes = phx.discretization.LevelSetRigidShapeSetPlan(
        (template,),
        np.zeros(2, dtype=np.int32),
        np.zeros(2, dtype=np.int32),
        maximum_contact_routes=2 * nodes.shape[0],
        maximum_candidate_slots=2 * nodes.shape[0],
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([17, 41], dtype=jnp.int64),
        jnp.ones(2, dtype=jnp.float64),
        ambient_dimension=3,
    ).prepare()
    material = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.4]]),
    )
    compiled = phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "level-set-grains", material, gravity=jnp.zeros(3)
        ),
        particles,
        shapes,
        phx.discretization.CompliantDEMMethodPlan(
            phx.discretization.DEMContactModelPlan(
                phx.discretization.LevelSetPenaltyNormalPlan(100.0)
            ),
            maximum_overlap_fraction=0.5,
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
        orientation=jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
    )
    result = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-4),
        None,
    )
    if not bool(result.successful):
        raise RuntimeError("Level-set DEM candidate refused; inspect detailed evidence.")
    print("accepted forces (N):", np.asarray(result.accepted_state.loads.total.force))
    print(
        "voxel relative volume error bound:",
        float(template.evidence.relative_volume_error_bound),
    )


if __name__ == "__main__":
    main()

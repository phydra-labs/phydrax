#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


axis = jnp.linspace(-1.0, 1.0, 8)
x, y = jnp.meshgrid(axis, axis, indexing="ij")
points = jnp.stack((x.reshape(-1), y.reshape(-1)), axis=1)
boundary = jnp.isclose(jnp.abs(points[:, 0]), 1.0) | jnp.isclose(
    jnp.abs(points[:, 1]), 1.0
)
normals = jnp.zeros_like(points)
normals = normals.at[:, 0].set(
    jnp.where(jnp.isclose(jnp.abs(points[:, 0]), 1.0), points[:, 0], 0.0)
)
normals = normals.at[:, 1].set(
    jnp.where(jnp.isclose(jnp.abs(points[:, 1]), 1.0), points[:, 1], 0.0)
)
length = jnp.linalg.norm(normals, axis=1)
normals = normals / jnp.where(length > 0.0, length, 1.0)[:, None]
discretization = phx.discretization.PointCloudPlan(
    points,
    jnp.ones((points.shape[0],)) / points.shape[0],
    boundary_mask=boundary,
    boundary_normals=normals,
    degree=2,
    neighbor_count=14,
).prepare()
exact = 1.0 - points[:, 0] ** 2 - points[:, 1] ** 2
source = phx.discretization.DissipativePointDiffusion(discretization).mv(exact)
boundary_values = jnp.where(boundary, exact, 0.0)
result = phx.discretization.solve_point_cloud_poisson(
    discretization,
    source,
    phx.discretization.PointBoundaryPlan("dirichlet", boundary_values),
)
print("residual", float(result.residual_norm))
print("maximum error", float(jnp.max(jnp.abs(result.values - exact))))
print("stencil condition", discretization.report.maximum_condition_number)

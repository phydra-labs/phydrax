# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Intrinsic sphere/torus Laplace--Beltrami against independent analytic fields.

The sampled branch supplies orientation only, not curvature or true projection.
Area-normalized density is explicit. Surface derivatives use tangent GMLS; the
paired divergence is separately conservative and is not mislabeled pointwise.
The ring-torus sampling needs at least 256 sites and eight local neighbors:
coarser clouds do not resolve a single tangent graph under the stated fit and
orientation bounds. Candidate discovery has explicit capacity and chunking.
"""

from __future__ import annotations

import argparse
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization.meshfree._stencils import LocalStencilPolicy
from phydrax.discretization.meshfree._surface import SurfacePointCloudPlan
from phydrax.discretization.meshfree._surface_geometry import (
    ImplicitSurfaceGeometry,
    SampledSurfaceGeometry,
)
from phydrax.discretization.meshfree._surface_quadrature import SurfaceQuadraturePolicy
from phydrax.metrix._ambient import RegularLevelSetManifold


WorkflowMetric = float | int | bool | str


def sphere_points(size: int, seed: int = 0) -> Array:
    if isinstance(size, bool) or not isinstance(size, int) or size < 32:
        raise ValueError("Surface workflow requires integer total capacity >=32.")
    index = np.arange(size, dtype=np.float64)
    z = 1 - 2 * (index + 0.5) / size
    azimuth = index * math.pi * (3 - math.sqrt(5)) + np.random.default_rng(seed).uniform(
        0, 2 * math.pi
    )
    radius = np.sqrt(1 - z * z)
    return jnp.asarray(
        np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))
    )


def surface_plan(
    *, size: int, seed: int = 0, neighbors: int = 24, chunk_rows: int = 32
) -> SurfacePointCloudPlan:
    points = sphere_points(size, seed)

    def constraint(point: Array) -> Array:
        return jnp.asarray([jnp.dot(point, point) - 1])

    source = RegularLevelSetManifold(
        constraint, ambient_dimension=3, codimension=1, manifold_id="unit-sphere"
    )
    quadrature = SurfaceQuadraturePolicy(
        "normalized-density",
        density=jnp.ones(size, dtype=points.dtype),
        total_area=4 * math.pi,
    )
    return SurfacePointCloudPlan(
        points,
        ImplicitSurfaceGeometry(
            source, certified_tube_radius=0.5, geometry_id="unit-sphere"
        ),
        neighbors=min(neighbors, size),
        quadrature=quadrature,
        stencil_policy=LocalStencilPolicy(polynomial_degree=2, chunk_rows=chunk_rows),
    )


def _relative_error(actual: Array, expected: Array, measures: Array) -> float:
    square = (actual - expected) ** 2
    reference = expected**2
    if actual.ndim == 2:
        square, reference = jnp.sum(square, axis=-1), jnp.sum(reference, axis=-1)
    return float(jnp.sqrt(jnp.sum(measures * square) / jnp.sum(measures * reference)))


def run_workflow(
    *, size: int, dimension: int = 3, seed: int = 0
) -> dict[str, WorkflowMetric]:
    if dimension != 3:
        raise ValueError("Closed-surface workflow requires ambient dimension 3.")
    if size < 256:
        raise ValueError(
            "Ring-torus tangent graph qualification requires total capacity >=256; use surface_plan for smaller sphere-only workloads."
        )
    neighbors = min(24, max(12, size // 8))
    exact = surface_plan(size=size, seed=seed, neighbors=neighbors).prepare()
    sampled = SurfacePointCloudPlan(
        exact.points,
        SampledSurfaceGeometry(
            reference_normals=exact.points, geometry_id="sampled-unit-sphere"
        ),
        neighbors,
        quadrature=exact.plan.quadrature,
        stencil_policy=exact.plan.stencil_policy,
    ).prepare()
    field = exact.points[:, 2]
    sphere_reference = -2 * field
    gradient_reference = (
        jnp.asarray([0, 0, 1], dtype=field.dtype)[None, :] - field[:, None] * exact.points
    )
    sphere_exact_error = _relative_error(
        exact.laplace_beltrami.mv(field), sphere_reference, exact.measures
    )
    sphere_sampled_error = _relative_error(
        sampled.laplace_beltrami.mv(field), sphere_reference, sampled.measures
    )
    # Irrational two-angle sampling preserves exactly the requested capacity;
    # the density of uniform parameter samples is 1/[r(R+r cos(theta))].
    index = np.arange(size, dtype=np.float64) + 0.5
    theta = 2 * math.pi * index / size
    phi = 2 * math.pi * np.mod(index * (math.sqrt(5) - 1) / 2, 1) + seed * 0.123
    major, minor = 2.0, 0.75
    radial = major + minor * np.cos(theta)
    torus_points = jnp.asarray(
        np.column_stack(
            (radial * np.cos(phi), radial * np.sin(phi), minor * np.sin(theta))
        )
    )
    torus_normals = jnp.asarray(
        np.column_stack(
            (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta))
        )
    )

    def constraint(point: Array) -> Array:
        return jnp.asarray(
            [
                (jnp.sqrt(point[0] ** 2 + point[1] ** 2) - major) ** 2
                + point[2] ** 2
                - minor**2
            ]
        )

    source = RegularLevelSetManifold(
        constraint, ambient_dimension=3, codimension=1, manifold_id="ring-torus"
    )
    density = jnp.asarray(1 / (minor * radial))
    quadrature = SurfaceQuadraturePolicy(
        "normalized-density", density=density, total_area=4 * math.pi**2 * major * minor
    )
    policy = LocalStencilPolicy(polynomial_degree=2, chunk_rows=32)
    torus_neighbors = 8
    torus_exact = SurfacePointCloudPlan(
        torus_points,
        ImplicitSurfaceGeometry(
            source, certified_tube_radius=0.25, geometry_id="ring-torus"
        ),
        torus_neighbors,
        quadrature=quadrature,
        stencil_policy=policy,
        maximum_candidates=size,
        target_chunk_size=32,
    ).prepare()
    torus_sampled = SurfacePointCloudPlan(
        torus_points,
        SampledSurfaceGeometry(
            reference_normals=torus_normals, geometry_id="sampled-ring-torus"
        ),
        torus_neighbors,
        quadrature=quadrature,
        stencil_policy=policy,
        maximum_candidates=size,
        target_chunk_size=32,
    ).prepare()
    torus_field = torus_points[:, 2]
    # Delta_s(r sin(theta)) = -sin(theta)/r - sin(theta)cos(theta)/(R+r cos(theta)).
    torus_reference = jnp.asarray(
        -np.sin(theta) / minor - np.sin(theta) * np.cos(theta) / radial
    )
    torus_exact_error = _relative_error(
        torus_exact.laplace_beltrami.mv(torus_field),
        torus_reference,
        torus_exact.measures,
    )
    torus_sampled_error = _relative_error(
        torus_sampled.laplace_beltrami.mv(torus_field),
        torus_reference,
        torus_sampled.measures,
    )
    tangent_vector = gradient_reference
    arrays = {
        id(leaf): leaf
        for leaf in jax.tree.leaves((exact, sampled, torus_exact, torus_sampled))
        if isinstance(leaf, Array)
    }
    retained_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in arrays.values())
    return {
        "capacity": size,
        "dimension": dimension,
        "domain": "unit sphere and ring torus (R=2,r=0.75)",
        "oracle_provenance": "independent Delta_S2 z=-2z; Delta_torus(r sin theta)=-sin(theta)/r-sin(theta)cos(theta)/(R+r cos(theta)); analytic radial normals",
        "retained_bytes": retained_bytes,
        "minimum_supported_capacity": 256,
        "torus_neighbors": torus_neighbors,
        "maximum_candidates": size,
        "target_chunk_size": 32,
        "sphere_exact_error": sphere_exact_error,
        "sphere_sampled_error": sphere_sampled_error,
        "torus_exact_error": torus_exact_error,
        "torus_sampled_error": torus_sampled_error,
        "sphere_exact_normal_error": float(
            jnp.max(jnp.linalg.norm(exact.normals - exact.points, axis=-1))
        ),
        "torus_exact_normal_error": float(
            jnp.max(jnp.linalg.norm(torus_exact.normals - torus_normals, axis=-1))
        ),
        "sphere_normal_error": float(
            jnp.max(jnp.linalg.norm(sampled.normals - exact.points, axis=-1))
        ),
        "torus_normal_error": float(
            jnp.max(jnp.linalg.norm(torus_sampled.normals - torus_normals, axis=-1))
        ),
        "laplace_relative_error": sphere_exact_error,
        "gradient_relative_error": _relative_error(
            exact.surface_gradient.mv(field), gradient_reference, exact.measures
        ),
        "constant_residual": float(
            jnp.max(jnp.abs(exact.laplace_beltrami.mv(jnp.ones(size, dtype=field.dtype))))
        ),
        "divergence_integral_residual": float(
            jnp.abs(jnp.sum(exact.measures * exact.surface_divergence.mv(tangent_vector)))
        ),
        "area_relative_error": float(
            jnp.abs(jnp.sum(exact.measures) - 4 * math.pi) / (4 * math.pi)
        ),
        "geometry_valid": bool(
            jnp.all(exact.geometry_evidence.valid)
            & jnp.all(sampled.geometry_evidence.valid)
            & jnp.all(torus_exact.geometry_evidence.valid)
            & jnp.all(torus_sampled.geometry_evidence.valid)
        ),
        "tube_certified": exact.geometry_evidence.tube_certified,
        "sampled_tube_certified": sampled.geometry_evidence.tube_certified,
        "approximation": policy.approximation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(run_workflow(size=args.size, seed=args.seed), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()

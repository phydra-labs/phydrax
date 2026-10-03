# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Manufactured variable-coefficient Poisson solve with native meshfree multigrid."""

from __future__ import annotations

import argparse
from typing import TypedDict

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import logical_array_bytes


class MultilevelPoissonMetrics(TypedDict):
    size: int
    point_count: int
    dimension: int
    seed: int
    maximum_error: float
    relative_residual: float
    boundary_residual: float
    converged: bool
    linear_status: int
    linear_iterations: int
    level_sizes: tuple[int, ...]
    constant_reproduction_error: float
    linear_reproduction_error: float
    transfer_entries: int
    stopping_reason: str
    domain: str
    oracle_provenance: str
    retained_bytes: int


def run_workflow(
    *, size: int = 64, dimension: int = 2, seed: int = 0
) -> MultilevelPoissonMetrics:
    """Use at most ``size`` points, not ``size`` points per coordinate axis.

    Independent reference: u=1-|x|², k=2+0.1 sum(x), and
    -div(k grad(u))=2 d k+0.2 sum(x). The boundary trace is prescribed exactly.
    Runtime convergence and reproduction metrics are observed, not inferred.
    """
    if dimension != 2:
        raise ValueError("The multilevel Poisson workflow supports dimension=2 only.")
    if size < 32:
        raise ValueError(
            "At least 32 points are required for this two-dimensional workflow."
        )
    rng = np.random.default_rng(seed)
    boundary_count = max(8, int(np.ceil(np.sqrt(size))))
    angles = np.arange(boundary_count) * (2 * np.pi / boundary_count)
    boundary_points = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    interior_count = size - boundary_count
    interior_angles = rng.uniform(0, 2 * np.pi, interior_count)
    radii = 0.9 * np.sqrt(rng.uniform(0, 1, interior_count))
    interior_points = radii[:, None] * np.stack(
        (np.cos(interior_angles), np.sin(interior_angles)), axis=1
    )
    points = jnp.asarray(
        np.concatenate((boundary_points, interior_points)), dtype=jnp.float64
    )
    boundary = jnp.arange(size) < boundary_count
    normals = jnp.where(boundary[:, None], points, 0.0)
    cloud = phx.discretization.PointCloudPlan(
        points,
        jnp.full((size,), np.pi / size),
        boundary_mask=boundary,
        boundary_normals=normals,
        neighbors=min(size, 30),
        stencil=phx.discretization.meshfree.LocalStencilPolicy(
            approximation="phs-rbf-fd", polynomial_degree=3
        ),
    ).prepare()
    exact = 1.0 - jnp.sum(points * points, axis=1)
    diffusivity = 2.0 + 0.1 * jnp.sum(points, axis=1)
    source = 2.0 * dimension * diffusivity + 0.2 * jnp.sum(points, axis=1)
    boundary_plan = phx.discretization.PointBoundaryPlan(
        (
            phx.discretization.PointBoundaryCondition(
                "dirichlet",
                np.arange(boundary_count),
                exact[:boundary_count],
                label="ring",
            ),
        ),
        row_count=size,
    )
    plan = phx.discretization.PointCloudPoissonPlan(
        cloud, boundary_plan, form="collocated"
    )
    # The prepared hierarchy eliminates Dirichlet identity rows from coarsening;
    # the plan then selects GMRES with the native meshfree multigrid cycle.
    hierarchy = plan.hierarchy_plan().prepare(plan.solve_space)
    poisson = phx.discretization.PointCloudPoissonPlan(
        cloud, boundary_plan, form="collocated", hierarchy=hierarchy
    ).prepare(diffusivity)
    result = poisson.solve(source)
    if poisson.hierarchy is None:
        raise RuntimeError("The multilevel consumer did not retain hierarchy evidence.")
    evidence = poisson.hierarchy.evidence
    physical_source = jnp.where(boundary, exact, source)
    relative_residual = float(
        result.residual_norm / jnp.maximum(jnp.linalg.norm(physical_source), 1e-30)
    )
    return {
        "size": size,
        "point_count": points.shape[0],
        "dimension": dimension,
        "seed": seed,
        "maximum_error": float(jnp.max(jnp.abs(result.values - exact))),
        "relative_residual": relative_residual,
        "boundary_residual": float(result.boundary_residual_norm),
        "converged": int(result.status) == int(phx.linalg.LinearSolveStatus.SUCCESS),
        "linear_status": int(result.status),
        "linear_iterations": int(result.diagnostics.iterations),
        "level_sizes": evidence.level_sizes,
        "constant_reproduction_error": max(
            (level[0] for level in evidence.reproduction_residuals), default=0.0
        ),
        "linear_reproduction_error": max(
            (level[1] for level in evidence.reproduction_residuals), default=0.0
        ),
        "transfer_entries": evidence.transfer_entries,
        "stopping_reason": evidence.stopping_reason,
        "domain": "unit disk in R^2; ring boundary and seeded interior points",
        "oracle_provenance": "independent differentiation: u=1-|x|^2, k=2+0.1 sum(x), f=2*d*k+0.2 sum(x)",
        "retained_bytes": logical_array_bytes(poisson),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name, value in run_workflow(
        size=args.size, dimension=args.dimension, seed=args.seed
    ).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()

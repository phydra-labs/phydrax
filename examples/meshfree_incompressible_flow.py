# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Incompressible meshfree flow: periodic Taylor–Green and bounded Stokes.

The periodic workflow advances the 2-D Taylor–Green vortex
``u = e^{-2 nu t} (sin x cos y, -cos x sin y)`` on a periodic point lattice whose
axial radius graph carries an admitted positive exterior metric. Each native
fixed step transports momentum conservatively, diffuses it implicitly, and
projects it with the compatible pressure owner; the ledger reports graph and
nodal divergence, kinetic energy against the exact decay ``e^{-4 nu t}``,
momentum, and artificial cycle content. A random graph-solenoidal edge field
is classified as non-physical and a CFL-violating step is refused and rolled
back before the run continues. The bounded workflow solves the manufactured
Stokes problem ``u = (sin(pi x) cos(pi y), -cos(pi x) sin(pi y))``,
``p = cos(pi x) cos(pi y)`` with the stabilized mixed owner.
"""

from __future__ import annotations

import argparse
import json
from typing import assert_never

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization import (
    MortonAddressPlan,
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MeshfreeEdgeRelationPlan,
    MeshfreeExteriorCalculusPlan,
    MeshfreeGeneralizedStokesPlan,
    MeshfreeMetricPolicy,
    minimum_image_edge_charts,
)
from phydrax.domain import HyperRectangle, PeriodicIdentification
from phydrax.metrix import EuclideanStateGeometry
from phydrax.solver import (
    CompatibleIncompressibleProjection,
    CompatibleVariableDensityProjection,
    FixedStepProblem,
    MeshfreeFlowEvidence,
    MeshfreeFlowStatus,
    MeshfreeIncompressibleFlowPlan,
    PreparedMeshfreeIncompressibleFlow,
    solve_fixed_step,
)


WorkflowMetric = float | int | bool | str | list[float]

LENGTH = 2.0 * np.pi


def periodic_flow(
    *, count: int = 12, viscosity: float = 0.05
) -> PreparedMeshfreeIncompressibleFlow:
    spacing = LENGTH / count
    lattice = np.stack(
        np.meshgrid(np.arange(count), np.arange(count), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    points = (lattice + 0.5) * spacing
    torus = HyperRectangle(np.zeros(2), np.full(2, LENGTH))
    address = MortonAddressPlan.from_periodic_identifications(
        tuple(PeriodicIdentification(torus, "x", component=axis) for axis in range(2)),
        maximum_depth=12,
    )
    capacity = 4 * count * count
    relation = MeshfreeEdgeRelationPlan(
        points, 1.1 * spacing, capacity, address=address
    ).prepare()
    exterior = MeshfreeExteriorCalculusPlan(
        points,
        1.1 * spacing,
        capacity,
        node_volumes=np.full(count * count, spacing**2),
        intrinsic_displacements=minimum_image_edge_charts(relation, points),
        metric_policy=MeshfreeMetricPolicy(),
    ).prepare(edge_relation=relation)
    cloud = PointCloudPlan(
        np.asarray(exterior.points),
        np.asarray(exterior.node_volumes),
        address=address,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()
    return MeshfreeIncompressibleFlowPlan(
        exterior,
        cloud,
        viscosity=viscosity,
        domain_betti_number=2,
        transport_scheme="reconstructed",
        cycle_tolerance=0.15,
    ).prepare()


def taylor_green(points: Array, time: float, viscosity: float) -> Array:
    x, y = points[:, 0], points[:, 1]
    return np.exp(-2.0 * viscosity * time) * jnp.stack(
        (jnp.sin(x) * jnp.cos(y), -jnp.cos(x) * jnp.sin(y)), axis=1
    )


def run_periodic(*, count: int = 12, steps: int = 10) -> dict[str, WorkflowMetric]:
    viscosity, step_size = 0.05, 0.1
    flow = periodic_flow(count=count, viscosity=viscosity)
    points = flow.points
    perturbed = taylor_green(points, 0.0, viscosity) + 0.1 * jnp.cos(
        points[:, 0] + points[:, 1]
    )[:, None] * jnp.ones((1, 2))
    initial = flow.initialize(perturbed)
    problem = FixedStepProblem(
        flow,
        initial.state,
        t0=0.0,
        t1=steps * step_size,
        step_size=step_size,
        state_geometry=EuclideanStateGeometry(),
    )
    solution = solve_fixed_step(problem, evidence_retention="steps")
    retained = solution.evidence
    if retained is None:
        raise RuntimeError("The periodic workflow requires retained fixed-step evidence.")
    evidence = retained.steps
    if not isinstance(evidence, MeshfreeFlowEvidence):
        raise RuntimeError("The periodic workflow requires meshfree flow step evidence.")
    energy = np.asarray(evidence.kinetic_energy_after) / float(
        evidence.kinetic_energy_before[0]
    )
    exact = np.exp(-4.0 * viscosity * step_size * np.arange(1, steps + 1))
    final = jax.tree.map(lambda leaf: leaf[-1], solution.states)
    # Refusal: a step far beyond the CFL limit is refused and rolled back; the
    # run then continues from the committed state.
    refused = flow.advance(final, steps * step_size, 50.0)
    resumed = flow.advance(refused.state, steps * step_size, step_size)
    noise = jnp.asarray(
        np.random.default_rng(1).standard_normal(flow.plan.exterior.lengths.shape)
    )
    match flow.plan.projection:
        case CompatibleIncompressibleProjection() as projection:
            projected = projection.project(noise)
        case CompatibleVariableDensityProjection() as projection:
            projected = projection.project(noise, final.density)
        case unknown:
            assert_never(unknown)
    cycles = flow.classify_edge_velocity(projected.candidate_velocity)
    return {
        "nodes": int(points.shape[0]),
        "cycle_space_dimension": flow.plan.cycle_space_dimension,
        "domain_betti_number": flow.plan.domain_betti_number,
        "initial_status": int(initial.evidence.status),
        "initial_divergence_before": float(initial.evidence.nodal_divergence_before),
        "initial_divergence_after": float(initial.evidence.nodal_divergence_after),
        "steps_accepted": int(jnp.sum(evidence.successful)),
        "max_graph_divergence": float(jnp.max(evidence.graph_divergence_norm)),
        "max_nodal_divergence": float(jnp.max(evidence.nodal_divergence_after)),
        "kinetic_energy_ratio": energy.tolist(),
        "exact_energy_ratio": exact.tolist(),
        "max_energy_relative_error": float(np.max(np.abs(energy / exact - 1.0))),
        "max_momentum": float(jnp.max(jnp.abs(evidence.momentum_after))),
        "final_velocity_error": float(
            jnp.max(
                jnp.abs(
                    final.velocity - taylor_green(points, steps * step_size, viscosity)
                )
            )
        ),
        "max_pressure_iterations": int(jnp.max(evidence.pressure_iterations)),
        "smooth_cycle_fraction": float(jnp.max(evidence.cycles.nonphysical_fraction)),
        "noise_cycle_fraction": float(cycles.nonphysical_fraction),
        "noise_classified_physical": bool(cycles.physical),
        "refused_status": MeshfreeFlowStatus(int(refused.evidence.status)).name,
        "refused_state_unchanged": bool(
            jnp.all(refused.state.velocity == final.velocity)
        ),
        "resumed_status": MeshfreeFlowStatus(int(resumed.evidence.status)).name,
    }


def stokes_velocity(x: Array) -> Array:
    return jnp.stack(
        (
            jnp.sin(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]),
            -jnp.cos(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
        )
    )


def stokes_pressure(x: Array) -> Array:
    return jnp.cos(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1])


def run_stokes(*, side: int = 13) -> dict[str, WorkflowMetric]:
    grid = np.meshgrid(*(np.linspace(0.0, 1.0, side),) * 2, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in grid], axis=1)
    boundary = np.any(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    points[~boundary] += np.random.default_rng(7).uniform(
        -0.15, 0.15, (np.count_nonzero(~boundary), 2)
    ) / (side - 1)
    normals = np.where(np.isclose(points, 1.0), 1.0, 0.0) - np.where(
        np.isclose(points, 0.0), 1.0, 0.0
    )
    cloud = PointCloudPlan(
        points,
        np.full(points.shape[0], 1.0 / points.shape[0]),
        boundary_mask=boundary,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()
    x = cloud.points
    exact = jax.vmap(stokes_velocity)(x)
    pressure = jax.vmap(stokes_pressure)(x)

    def force(y: Array) -> Array:
        laplacian = jnp.trace(
            jax.jacfwd(jax.jacfwd(stokes_velocity))(y), axis1=1, axis2=2
        )
        return -laplacian + jax.grad(stokes_pressure)(y)

    rows = np.flatnonzero(boundary)
    plan = MeshfreeGeneralizedStokesPlan(
        cloud,
        PointBoundaryPlan(
            tuple(
                PointBoundaryCondition(
                    "dirichlet", rows, exact[rows, a], label=f"wall-{a}", component=a
                )
                for a in range(2)
            ),
            row_count=x.shape[0],
            components=2,
        ),
        shear_modulus=1.0,
    )
    result = plan.prepare().solve(jax.vmap(force)(x))
    weights = cloud.quadrature_weights
    shifted = pressure - jnp.sum(weights * pressure) / jnp.sum(weights)
    return {
        "nodes": int(x.shape[0]),
        "status": int(result.status),
        "linear_iterations": int(result.linear.diagnostics.iterations),
        "velocity_rms_error": float(
            jnp.sqrt(jnp.sum(weights[:, None] * (result.field - exact) ** 2))
        ),
        "pressure_rms_error": float(
            jnp.sqrt(jnp.sum(weights * (result.pressure - shifted) ** 2))
        ),
        "max_divergence": float(jnp.max(jnp.abs(result.divergence))),
        "pressure_oscillation": float(result.pressure_oscillation),
        "gauge_compatibility_residual": float(result.compatibility_residual),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--side", type=int, default=13)
    args = parser.parse_args()
    report = {
        "periodic_taylor_green": run_periodic(count=args.count, steps=args.steps),
        "bounded_stokes": run_stokes(side=args.side),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

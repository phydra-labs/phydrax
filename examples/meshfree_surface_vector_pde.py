# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Tangential vector PDEs on the unit sphere with independent analytic oracles.

For a trace-free quadratic ``Y = x^T C x`` (a degree-2 spherical harmonic),
``grad_S Y`` and ``n x grad_S Y`` are vector spherical harmonics: Hodge
eigenvalue ``-6`` and, since ``Ric = g`` on the unit sphere, Bochner eigenvalue
``-5``. The example checks both operators, their Weitzenboeck difference, a
tangency-constrained Hodge-Helmholtz saddle solve and a closed-surface
Stokes/Brinkman solve whose exact velocity is divergence free and whose exact
pressure is the degree-1 harmonic ``z``.
"""

from __future__ import annotations

import argparse
import json
import math

import jax.numpy as jnp
from jax import Array

import phydrax.linalg as la
from examples.meshfree_surface_laplace_beltrami import sphere_points
from phydrax.discretization.meshfree import (
    ImplicitSurfaceGeometry,
    LocalStencilPolicy,
    SurfacePointCloudPlan,
    SurfaceQuadraturePolicy,
    SurfaceTangentCalculus,
)
from phydrax.ein import contract
from phydrax.metrix import RegularLevelSetManifold


WorkflowMetric = float | int | bool | str

HARMONIC = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.5], [0.0, 0.5, 0.0]])


def sphere_calculus(
    *, size: int = 400, neighbors: int = 30, degree: int = 4, seed: int = 0
) -> SurfaceTangentCalculus:
    def constraint(point: Array) -> Array:
        return jnp.asarray([jnp.dot(point, point) - 1])

    source = RegularLevelSetManifold(
        constraint, ambient_dimension=3, codimension=1, manifold_id="unit-sphere"
    )
    prepared = SurfacePointCloudPlan(
        sphere_points(size, seed),
        ImplicitSurfaceGeometry(source, geometry_id="unit-sphere"),
        neighbors,
        quadrature=SurfaceQuadraturePolicy("tangent-voronoi"),
        stencil_policy=LocalStencilPolicy(polynomial_degree=degree, chunk_rows=128),
    ).prepare()
    return SurfaceTangentCalculus(prepared)


def harmonic_fields(points: Array, normals: Array) -> tuple[Array, Array, Array]:
    """``(Y, grad_S Y, n x grad_S Y)`` for the trace-free quadratic harmonic."""
    value = jnp.sum(points * (points @ HARMONIC), axis=-1)
    ambient = 2 * points @ HARMONIC
    gradient = ambient - jnp.sum(ambient * normals, axis=-1, keepdims=True) * normals
    return value, gradient, jnp.cross(normals, gradient)


def _relative(actual: Array, expected: Array, measures: Array) -> float:
    return float(
        jnp.sqrt(
            jnp.sum(measures[:, None] * (actual - expected) ** 2)
            / jnp.sum(measures[:, None] * expected**2)
        )
    )


def _policy(relative: float) -> la.LinearSolvePolicy:
    return la.LinearSolvePolicy(
        la.GMRES(restart=400),
        tolerance=la.TolerancePolicy(relative=relative, absolute=1e-14, max_steps=3000),
    )


def run_workflow(*, size: int = 400) -> dict[str, WorkflowMetric]:
    calculus = sphere_calculus(size=size)
    surface = calculus.surface
    points, normals, measures = surface.points, surface.normals, surface.measures
    _, gradient, rotated = harmonic_fields(points, normals)
    metrics: dict[str, WorkflowMetric] = {"nodes": size}
    for name, field in (("gradient", gradient), ("rotated", rotated)):
        hodge = calculus.hodge_laplacian.mv(field)
        bochner = calculus.bochner_laplacian.mv(field)
        metrics[f"{name}_hodge_error"] = _relative(hodge, -6 * field, measures)
        metrics[f"{name}_bochner_error"] = _relative(bochner, -5 * field, measures)
        metrics[f"{name}_weitzenboeck_error"] = _relative(
            bochner - hodge,
            contract("nab,nb->na", calculus.ricci, field),
            measures,
        )
    helmholtz = calculus.tangent_vector_system("hodge", diffusivity=1.0, reaction=1.0)
    helmholtz_result = la.solve(
        helmholtz, calculus.tangent_rhs(7 * gradient), policy=_policy(1e-10)
    )
    metrics["helmholtz_successful"] = bool(helmholtz_result.successful)
    metrics["helmholtz_error"] = _relative(helmholtz_result.value[0], gradient, measures)
    metrics["helmholtz_tangency"] = float(
        jnp.max(jnp.abs(calculus.normal_constraint.mv(helmholtz_result.value[0])))
    )
    pressure = points[:, 2]
    pressure_gradient = jnp.asarray([0.0, 0.0, 1.0]) - pressure[:, None] * normals
    reaction = 1.0
    stokes = calculus.stokes_system(viscosity=1.0, reaction=reaction)
    stokes_result = la.solve(
        stokes,
        calculus.stokes_rhs((reaction + 5) * rotated + pressure_gradient),
        policy=_policy(1e-8),
    )
    velocity, computed_pressure = stokes_result.value[0], stokes_result.value[1][:, 0]
    exact_pressure = pressure - jnp.sum(measures * pressure) / jnp.sum(measures)
    metrics.update(
        {
            "stokes_successful": bool(stokes_result.successful),
            "stokes_iterations": int(stokes_result.diagnostics.iterations),
            "stokes_velocity_error": _relative(velocity, rotated, measures),
            "stokes_pressure_error": float(
                jnp.sqrt(
                    jnp.sum(measures * (computed_pressure - exact_pressure) ** 2)
                    / jnp.sum(measures * exact_pressure**2)
                )
            ),
            "stokes_divergence": float(
                jnp.max(jnp.abs(calculus.divergence.mv(velocity)))
            ),
            "stokes_tangency": float(
                jnp.max(jnp.abs(calculus.normal_constraint.mv(velocity)))
            ),
            "stokes_gauge_residual": float(
                calculus.stokes_gauge_residual(stokes_result.value)
            ),
            "gauss_curvature_error": float(
                jnp.max(jnp.abs(calculus.gauss_curvature - 1))
            ),
            "area_error": abs(
                float(surface.quadrature_evidence.total_area) - 4 * math.pi
            ),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(run_workflow(size=args.size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

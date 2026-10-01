#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
from typing import Literal, TypeAlias

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import logical_array_bytes
from phydrax.discretization import PointBoundaryKind, PointDiffusionForm
from phydrax.discretization.meshfree import LocalStencilPolicy
from phydrax.linalg import SparseAssemblyPolicy
from phydrax.typing import parse


ManufacturedSolution: TypeAlias = Literal["quadratic", "exponential"]


def run_workflow(
    *,
    size: int = 64,
    dimension: int = 2,
    seed: int = 0,
    manufactured: ManufacturedSolution = "quadratic",
    boundary_kind: PointBoundaryKind = "dirichlet",
    form: PointDiffusionForm = "collocated",
    stencil: LocalStencilPolicy | None = None,
    neighbors: int | None = None,
    target_chunk_size: int | None = None,
    assembly_policy: SparseAssemblyPolicy | None = None,
) -> dict[str, float | int | bool | str]:
    """Solve an independently manufactured variable-coefficient Poisson problem.

    ``size`` bounds total point capacity, not points per axis. Exponential data
    provide nonpolynomial refinement evidence; quadratic data probe exactness.
    Neumann explicitly projects incompatible discrete data, reports the source
    correction, and compares the solution with the point-gauged exact field.
    Dissipative results are observed errors, not an accuracy certification.

    The demonstrated accuracy route is cubic-augmented PHS-RBF-FD with the
    native support-count policy, uniformly for every boundary kind. Local
    GMLS acceptance alone is not a global Neumann stability certificate.
    A supplied native stencil policy remains explicit and is reported, with
    unchanged algebraic/physical refusal checks rather than hidden fallback.
    """
    manufactured = parse(manufactured, ManufacturedSolution, "manufactured")
    boundary_kind = parse(boundary_kind, PointBoundaryKind, "boundary_kind")
    form = parse(form, PointDiffusionForm, "form")
    if dimension not in (1, 2, 3):
        raise ValueError("Poisson example supports dimensions one through three.")
    if size < 4**dimension:
        raise ValueError("Capacity must allow at least four points per axis.")
    side = int(size ** (1.0 / dimension))
    while (side + 1) ** dimension <= size:
        side += 1
    while side**dimension > size:
        side -= 1
    coordinates = np.meshgrid(
        *([np.linspace(-1.0, 1.0, side)] * dimension), indexing="ij"
    )
    points_host = np.stack([coordinate.reshape(-1) for coordinate in coordinates], axis=1)
    boundary_host = np.any(np.isclose(np.abs(points_host), 1.0), axis=1)
    rng = np.random.default_rng(seed)
    points_host[~boundary_host] += (
        rng.uniform(-0.04, 0.04, (np.count_nonzero(~boundary_host), dimension)) / side
    )
    points = jnp.asarray(points_host)
    boundary = jnp.asarray(boundary_host)
    normals = jnp.where(jnp.isclose(jnp.abs(points), 1.0), points, 0.0)
    normals = normals / jnp.maximum(jnp.linalg.norm(normals, axis=1, keepdims=True), 1.0)
    count = points.shape[0]
    stencil_policy = (
        LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3)
        if stencil is None
        else stencil
    )
    face_count = jnp.sum(jnp.isclose(jnp.abs(points), 1.0), axis=1)
    spacing = 2.0 / (side - 1)
    boundary_weights = (
        spacing ** (dimension - 1)
        * jnp.sqrt(face_count)
        * 0.5 ** jnp.maximum(face_count - 1, 0)
    )
    discretization = phx.discretization.PointCloudPlan(
        points,
        jnp.full((count,), 2.0**dimension / count),
        boundary_mask=boundary,
        boundary_normals=normals,
        boundary_quadrature_weights=boundary_weights,
        stencil=stencil_policy,
        neighbors=neighbors,
        target_chunk_size=target_chunk_size,
    ).prepare()
    coordinate_sum = jnp.sum(points, axis=1)
    diffusivity = 2.0 + 0.1 * coordinate_sum
    if manufactured == "quadratic":
        exact = 1.0 - jnp.sum(points * points, axis=1)
        source = 2.0 * dimension * diffusivity + 0.2 * coordinate_sum
        exact_gradient = -2.0 * points
        oracle = "u=1-|x|², k=2+0.1 sum(x), f=2*d*k+0.2 sum(x)"
    else:
        exact = jnp.exp(coordinate_sum / dimension)
        source = -exact * (diffusivity / dimension + 0.1)
        exact_gradient = jnp.broadcast_to(exact[:, None] / dimension, points.shape)
        oracle = "u=exp(sum(x)/d), k=2+0.1 sum(x), f=-u*(k/d+0.1)"
    conormal = diffusivity * jnp.sum(exact_gradient * normals, axis=1)
    if boundary_kind == "dirichlet":
        boundary_plan = phx.discretization.PointBoundaryPlan("dirichlet", exact)
    elif boundary_kind == "neumann":
        boundary_plan = phx.discretization.PointBoundaryPlan("neumann", conormal)
    else:
        boundary_plan = phx.discretization.PointBoundaryPlan(
            "robin", conormal + exact, robin_coefficient=1.0
        )
    prepared = phx.discretization.PointCloudPoissonPlan(
        discretization,
        boundary_plan,
        form=form,
        preconditioner="ilu",
        compatibility="project" if boundary_kind == "neumann" else "refuse",
        assembly_policy=assembly_policy,
    ).prepare(diffusivity)
    result = prepared.solve(source)
    reference = (
        exact - exact[prepared.plan.gauge_index] if boundary_kind == "neumann" else exact
    )
    sbp = phx.discretization.point_sbp_report(discretization)
    scale = (
        discretization.quadrature_weights
        if form == "dissipative"
        else jnp.ones_like(source)
    )
    original_rhs = jnp.where(boundary, boundary_plan.values, scale * source)
    original_residual = (
        prepared.physical_assembly.operator.mv(result.values) - original_rhs
    )
    if boundary_kind == "dirichlet":
        original_residual = jnp.where(
            boundary, result.values - boundary_plan.values, original_residual
        )
    provenance = f"Independent analytical {oracle}; exact {boundary_kind} boundary data."
    if boundary_kind == "neumann":
        provenance += (
            " Analytic source supplied unchanged; native discrete source projection explicitly applied and reported;"
            f" continuum reference gauge-shifted at interior point {prepared.plan.gauge_index}."
        )
    return {
        "point_count": count,
        "declared_capacity": size,
        "spacing": spacing,
        "dimension": dimension,
        "approximation": discretization.plan.stencil.approximation,
        "polynomial_degree": discretization.plan.stencil.polynomial_degree,
        "neighbors": discretization.plan.neighbors,
        "domain": f"[-1,1]^{dimension} perturbed Cartesian cloud",
        "oracle_provenance": provenance,
        "retained_bytes": logical_array_bytes(prepared),
        "assembly_symbolic_workspace_bytes": prepared.assembly.plan.cost.symbolic_workspace_bytes,
        "assembly_numeric_workspace_bytes": prepared.assembly.plan.cost.numeric_workspace_bytes,
        "physical_assembly_symbolic_workspace_bytes": prepared.physical_assembly.plan.cost.symbolic_workspace_bytes,
        "physical_assembly_numeric_workspace_bytes": prepared.physical_assembly.plan.cost.numeric_workspace_bytes,
        "maximum_solution_error": float(jnp.max(jnp.abs(result.values - reference))),
        "maximum_boundary_error": float(
            jnp.max(jnp.where(boundary, jnp.abs(result.values - reference), 0.0))
        ),
        "manufactured": manufactured,
        "boundary_kind": boundary_kind,
        "form": form,
        "compatibility_policy": prepared.plan.compatibility,
        "maximum_source_correction": float(jnp.max(jnp.abs(result.source_correction))),
        "source_correction_l2": float(jnp.linalg.norm(result.source_correction)),
        "original_compatibility_residual": float(result.compatibility_residual),
        "true_original_residual": float(jnp.linalg.norm(original_residual)),
        "gauge_residual": float(result.gauge_residual),
        "sbp_passed": sbp.passed,
        "sbp_green_residual": sbp.maximum_green_residual,
        "sbp_conservation_residual": sbp.maximum_conservation_residual,
        "residual_norm": float(result.residual_norm),
        "residual_tolerance": float(result.residual_tolerance),
        "boundary_residual_norm": float(result.boundary_residual_norm),
        "linear_status": int(result.status),
        "linear_iterations": int(result.diagnostics.iterations),
        "compatible": bool(result.compatible),
        "successful": bool(result.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--manufactured", choices=("quadratic", "exponential"), default="quadratic"
    )
    parser.add_argument(
        "--boundary-kind", choices=("dirichlet", "neumann", "robin"), default="dirichlet"
    )
    parser.add_argument(
        "--form", choices=("collocated", "dissipative"), default="collocated"
    )
    args = parser.parse_args()
    for name, value in run_workflow(
        size=args.size,
        dimension=args.dimension,
        seed=args.seed,
        manufactured=args.manufactured,
        boundary_kind=args.boundary_kind,
        form=args.form,
    ).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()

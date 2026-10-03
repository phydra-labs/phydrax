#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import math
from typing import assert_never, get_args, Literal, TypeAlias, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

import phydrax as phx
from benchmarks._runtime import logical_array_bytes
from phydrax.discretization import PointBoundaryKind, PointDiffusionForm
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    PointGhostLayerPlan,
    prepare_tensor_point_sbp,
)
from phydrax.linalg import SparseAssemblyPolicy
from phydrax.typing import parse


ManufacturedSolution: TypeAlias = Literal["quadratic", "exponential"]
ManufacturedBoundary: TypeAlias = Literal["dirichlet", "neumann", "robin", "mixed"]


class PoissonWorkflowReport(TypedDict):
    point_count: int
    declared_capacity: int
    spacing: float
    fill_distance: float
    fill_distance_provenance: str
    separation_distance: float
    dimension: int
    approximation: str
    polynomial_degree: int
    neighbors: int
    domain: str
    quadrature: str
    oracle_provenance: str
    retained_bytes: int
    assembly_symbolic_workspace_bytes: int
    assembly_numeric_workspace_bytes: int
    physical_assembly_symbolic_workspace_bytes: int
    physical_assembly_numeric_workspace_bytes: int
    maximum_solution_error: float
    maximum_boundary_error: float
    manufactured: ManufacturedSolution
    boundary_kind: ManufacturedBoundary
    boundary_route: str
    ghost_extension_defect: float
    form: PointDiffusionForm
    compatibility_policy: str
    maximum_source_correction: float
    source_correction_l2: float
    original_compatibility_residual: float
    true_original_residual: float
    original_load_norm: float
    gauge_residual: float
    sbp_passed: bool
    derivative_realization: str
    sbp_grid_id: str | None
    sbp_norm_id: str | None
    sbp_family_ids: tuple[str, ...]
    sbp_axis_names: tuple[str, ...]
    sbp_interior_order: int | None
    sbp_stable_realization: bool | None
    sbp_successful: bool | None
    sbp_reproduction_degree: int | None
    sbp_tolerance: float | None
    sbp_reproduction_residual: float | None
    sbp_green_residual: float
    sbp_conservation_residual: float
    continuum_consistent: bool
    residual_norm: float
    residual_tolerance: float
    boundary_residual_norm: float
    linear_status: int
    linear_iterations: int
    compatible: bool
    algebraically_successful: bool
    successful: bool


def run_workflow(
    *,
    size: int = 64,
    dimension: int = 2,
    seed: int = 0,
    manufactured: ManufacturedSolution = "quadratic",
    boundary_kind: ManufacturedBoundary = "dirichlet",
    form: PointDiffusionForm = "collocated",
    stencil: LocalStencilPolicy | None = None,
    neighbors: int | None = None,
    target_chunk_size: int | None = None,
    assembly_policy: SparseAssemblyPolicy | None = None,
) -> PoissonWorkflowReport:
    """Solve an independently manufactured variable-coefficient Poisson problem.

    ``size`` bounds total point capacity, not points per axis. Exponential data
    provide nonpolynomial refinement evidence; quadratic data probe exactness.
    Neumann explicitly projects incompatible discrete data, reports the source
    correction, and compares the solution with the point-gauged exact field.
    ``mixed`` owns the ``x0=-1`` face (corners included) by Dirichlet rows, the
    remaining ``x0=+1`` face by Neumann rows, and every other face by Robin
    rows; each entity carries its physical normals and boundary measure.
    Dissipative execution explicitly binds a native point-primary TensorGrid,
    second-order SBPDerivativePlan families, and their SBPGridNorm to the cloud
    samples and physical volume/face cubature. This is a structured native
    tensor SBP realization, not an arbitrary-cloud meshfree stability claim.
    It requires float64; measured analytic errors remain separate evidence.

    The demonstrated collocated accuracy route is cubic-augmented PHS-RBF-FD
    with the native support-count policy, uniformly for every boundary kind.
    Collocated Neumann and Robin rows use the declared ghost-layer PDE+BC route
    (``PointGhostLayerPlan``): replacing the PDE at a boundary point by its
    flux condition alone yields spectrally unstable square collocation that
    the default stability assessment refuses. The dissipative form imposes its
    conormal data weakly and keeps the square route.
    A supplied native stencil policy remains explicit and is reported, with
    unchanged algebraic/physical refusal checks rather than hidden fallback.
    """
    manufactured = parse(manufactured, ManufacturedSolution, "manufactured")
    boundary_kind = parse(boundary_kind, ManufacturedBoundary, "boundary_kind")
    form = parse(form, PointDiffusionForm, "form")
    if dimension not in (1, 2, 3):
        raise ValueError("Poisson example supports dimensions one through three.")
    match form:
        case "collocated":
            minimum_side = 4
        case "dissipative":
            # The declared order-two native SBP closure requires five points.
            minimum_side = 5
        case unknown:
            assert_never(unknown)
    if size < minimum_side**dimension:
        raise ValueError(
            f"Capacity must allow at least {minimum_side} points per axis for {form}."
        )
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
    native_grid = None
    native_derivatives: tuple[phx.discretization.PreparedSBPOperator, ...] = ()
    native_norm = None
    if form == "collocated":
        rng = np.random.default_rng(seed)
        points_host[~boundary_host] += (
            rng.uniform(-0.04, 0.04, (np.count_nonzero(~boundary_host), dimension)) / side
        )
    else:
        if not jax.config.x64_enabled:
            raise ValueError(
                "The native tensor SBP workflow requires JAX float64 execution."
            )
        native_grid = phx.discretization.TensorGridPlan(
            tuple(phx.discretization.UniformAxisSpec(side) for _ in range(dimension)),
            axis_names=tuple(f"x{axis}" for axis in range(dimension)),
        ).prepare(jnp.asarray([[-1.0] * dimension, [1.0] * dimension], dtype=jnp.float64))
        native_derivatives = tuple(
            phx.discretization.SBPDerivativePlan(
                native_grid, axis, interior_order=2
            ).prepare()
            for axis in native_grid.axis_names
        )
        native_norm = phx.discretization.SBPGridNorm(native_derivatives)
        points_host = np.asarray(native_grid.points)
    points = jnp.asarray(points_host)
    boundary = jnp.asarray(boundary_host)
    normals = jnp.where(jnp.isclose(jnp.abs(points), 1.0), points, 0.0)
    normals = normals / jnp.maximum(jnp.linalg.norm(normals, axis=1, keepdims=True), 1.0)
    count = points.shape[0]
    stencil_policy = (
        (
            LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3)
            if form == "collocated"
            else LocalStencilPolicy(polynomial_degree=2)
        )
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
    volume_weights = jnp.full((count,), 2.0**dimension / count)
    if native_norm is not None:
        volume_weights = native_norm.weights.reshape((-1,))
        # Each face owns its tangential norm product. At corners the vector sum
        # preserves every signed SBP face measure in a single cloud boundary row.
        signed_face_weights = np.zeros(points_host.shape, dtype=np.float64)
        for axis in range(dimension):
            tangential = np.ones((side,) * dimension, dtype=np.float64)
            for tangent, factor in enumerate(native_norm.axis_weights):
                if tangent != axis:
                    shape = [1] * dimension
                    shape[tangent] = side
                    tangential *= np.asarray(factor).reshape(shape)
            signed_face_weights[:, axis] = (
                tangential.reshape((-1,))
                * points_host[:, axis]
                * np.isclose(np.abs(points_host[:, axis]), 1.0)
            )
        boundary_weights = jnp.asarray(np.linalg.norm(signed_face_weights, axis=1))
        normals = jnp.asarray(
            signed_face_weights
            / np.maximum(np.asarray(boundary_weights)[:, None], np.finfo(np.float64).tiny)
        )
        if neighbors is None:
            neighbors = min(count, 3 * math.comb(dimension + 2, 2))
    discretization = phx.discretization.PointCloudPlan(
        points,
        volume_weights,
        boundary_mask=boundary,
        boundary_normals=normals,
        point_ids=np.arange(count, dtype=np.int64) if native_grid is not None else None,
        boundary_quadrature_weights=boundary_weights,
        stencil=stencil_policy,
        neighbors=neighbors,
        target_chunk_size=target_chunk_size,
    ).prepare()
    sbp_derivatives = (
        prepare_tensor_point_sbp(discretization, native_derivatives)
        if form == "dissipative"
        else None
    )
    if sbp_derivatives is not None and not bool(sbp_derivatives.successful):
        raise RuntimeError(
            "The native tensor SBP binding was refused: "
            f"axis statuses {np.asarray(sbp_derivatives.status).tolist()}."
        )
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
    first_low = np.isclose(points_host[:, 0], -1.0)
    first_high = np.isclose(points_host[:, 0], 1.0) & ~first_low
    groups: tuple[tuple[PointBoundaryKind, np.ndarray], ...]
    match boundary_kind:
        case "dirichlet" | "neumann" | "robin":
            groups = ((boundary_kind, boundary_host),)
        case "mixed":
            groups = (
                ("dirichlet", first_low),
                ("neumann", first_high),
                ("robin", boundary_host & ~first_low & ~first_high),
            )
        case _:
            assert_never(boundary_kind)
    conditions = []
    for kind, mask in groups:
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            continue
        values = {
            "dirichlet": exact,
            "neumann": conormal,
            "robin": conormal + exact,
        }[kind][rows]
        conditions.append(
            phx.discretization.PointBoundaryCondition(
                kind,
                rows,
                values,
                label=f"{kind}-faces",
                normals=None if kind == "dirichlet" else normals[rows],
                measure=boundary_weights[rows],
                robin_coefficient=1.0 if kind == "robin" else None,
            )
        )
    boundary_plan = phx.discretization.PointBoundaryPlan(conditions, row_count=count)
    ghosts = (
        PointGhostLayerPlan(boundary_plan).prepare(discretization)
        if form == "collocated" and boundary_kind != "dirichlet"
        else None
    )
    prepared = phx.discretization.PointCloudPoissonPlan(
        discretization,
        boundary_plan,
        form=form,
        compatibility="project" if boundary_kind == "neumann" else "refuse",
        assembly_policy=assembly_policy,
        ghosts=ghosts,
        sbp=sbp_derivatives,
    ).prepare(diffusivity)
    result = prepared.solve(source)
    unknowns = (
        result.values
        if result.ghost_values is None
        else jnp.concatenate((result.values, result.ghost_values))
    )
    gauges = prepared.plan.gauges
    reference = exact - exact[gauges[0]] if gauges else exact
    if sbp_derivatives is None:
        local_sbp = phx.discretization.point_sbp_report(discretization)
        sbp_passed = local_sbp.passed
        sbp_green_residual = local_sbp.maximum_green_residual
        sbp_conservation_residual = local_sbp.maximum_conservation_residual
    else:
        sbp_passed = bool(sbp_derivatives.successful)
        sbp_green_residual = float(jnp.max(sbp_derivatives.green_residual))
        sbp_conservation_residual = float(jnp.max(sbp_derivatives.conservation_residual))
    # The true original residual uses the unprojected analytic source.
    original_load = prepared.physical_rhs(source)
    original_residual = prepared.physical_assembly.operator.mv(unknowns) - original_load
    provenance = f"Independent analytical {oracle}; exact {boundary_kind} boundary data."
    if gauges:
        provenance += (
            " Analytic source supplied unchanged; native discrete source projection explicitly applied and reported;"
            f" continuum reference gauge-shifted at interior point {gauges[0]}."
        )
    # Measured cloud geometry: the fill distance over scrambled Sobol probes
    # and the separation distance are what observed orders are gated against.
    tree = cKDTree(points_host)
    probes = (
        2.0
        * qmc.Sobol(d=dimension, scramble=True, seed=seed + 7919).random_base2(
            math.ceil(math.log2(4 * count))
        )
        - 1.0
    )
    fill_distance = float(np.max(tree.query(probes)[0]))
    separation_distance = 0.5 * float(np.min(tree.query(points_host, k=2)[0][:, 1]))
    return {
        "point_count": count,
        "declared_capacity": size,
        "spacing": spacing,
        "fill_distance": fill_distance,
        "fill_distance_provenance": "max nearest distance over scrambled Sobol probes of [-1,1]^d",
        "separation_distance": separation_distance,
        "dimension": dimension,
        "derivative_realization": (
            "native-tensor-sbp"
            if native_grid is not None
            else "local-meshfree-collocation"
        ),
        "sbp_grid_id": None if native_grid is None else native_grid.prepared_id,
        "sbp_norm_id": None if native_norm is None else native_norm.norm_id,
        "sbp_family_ids": tuple(
            derivative.plan.family.family_id for derivative in native_derivatives
        ),
        "sbp_axis_names": () if native_grid is None else native_grid.axis_names,
        "sbp_interior_order": None if native_grid is None else 2,
        "sbp_stable_realization": (
            None if sbp_derivatives is None else sbp_derivatives.stable_realization
        ),
        "approximation": discretization.plan.stencil.approximation,
        "polynomial_degree": discretization.plan.stencil.polynomial_degree,
        "neighbors": discretization.plan.neighbors,
        "domain": (
            f"[-1,1]^{dimension} perturbed Cartesian cloud"
            if form == "collocated"
            else f"[-1,1]^{dimension} native point-primary TensorGrid; C-order grid row map"
        ),
        "quadrature": (
            "equal volume weights; tensor-trapezoid face quadrature"
            if form == "collocated"
            else "SBPGridNorm volume; native tensor norm tangential face cubature"
        ),
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
        "boundary_route": prepared.plan.route,
        "ghost_extension_defect": (
            float("nan")
            if result.ghost_extension_defect is None
            else float(result.ghost_extension_defect)
        ),
        "form": form,
        "compatibility_policy": prepared.plan.compatibility,
        "maximum_source_correction": float(jnp.max(jnp.abs(result.source_correction))),
        "source_correction_l2": float(jnp.linalg.norm(result.source_correction)),
        "original_compatibility_residual": float(result.compatibility_residual),
        "true_original_residual": float(jnp.linalg.norm(original_residual)),
        "original_load_norm": float(jnp.linalg.norm(original_load)),
        "gauge_residual": float(result.gauge_residual),
        "sbp_passed": sbp_passed,
        "sbp_successful": (
            None if sbp_derivatives is None else bool(sbp_derivatives.successful)
        ),
        "sbp_reproduction_degree": (
            None if sbp_derivatives is None else sbp_derivatives.reproduction_degree
        ),
        "sbp_tolerance": None if sbp_derivatives is None else sbp_derivatives.tolerance,
        "sbp_reproduction_residual": (
            None
            if sbp_derivatives is None
            else float(jnp.max(sbp_derivatives.reproduction_residual))
        ),
        "sbp_green_residual": sbp_green_residual,
        "sbp_conservation_residual": sbp_conservation_residual,
        "continuum_consistent": result.continuum_consistent,
        "residual_norm": float(result.residual_norm),
        "residual_tolerance": float(result.residual_tolerance),
        "boundary_residual_norm": float(result.boundary_residual_norm),
        "linear_status": int(result.status),
        "linear_iterations": int(result.diagnostics.iterations),
        "compatible": bool(result.compatible),
        "algebraically_successful": bool(result.algebraically_successful),
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
        "--boundary-kind", choices=get_args(ManufacturedBoundary), default="dirichlet"
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

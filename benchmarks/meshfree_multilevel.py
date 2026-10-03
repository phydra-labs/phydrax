# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compare native ILU, smoothed aggregation, and meshfree hierarchy on one fine system."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import qmc

from benchmarks._io import write_json_atomic
from benchmarks._runtime import logical_array_bytes
from benchmarks.meshfree_scaling import (
    add_config_arguments,
    admitted_rows,
    apply_baseline,
    cloud_points,
    config_from_arguments,
    configure_precision,
    DeclaredCapacityRefusal,
    failure_record,
    make_record,
    MeshfreeConfig,
    PhaseRecorder,
)


# The dissipative GMLS form Dᵀ M D on stratified clouds has oscillatory
# near-null modes (a smoothing least-squares gradient barely sees them): at
# N=256 its smallest Dirichlet eigenvalue is ~200x below the PHS-RBF-FD one and
# no preconditioner is h-robust. The fine system therefore uses the admitted
# elliptic stencil of the Q2 campaign: PHS-RBF-FD degree 3 with 2x basis support.
_STENCIL_DEGREE = 3


def dirichlet_cube_cloud(
    capacity: int, dimension: int, seed: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified cube faces then interior; the faces are the convex hull.

    Each of the ``2 d`` faces carries ``capacity^((d-1)/d)`` stratified samples
    (quasi-uniform with the interior), so Dirichlet data holds on the whole
    hull rather than on its few extreme vertices.
    """
    per_face = round(capacity ** ((dimension - 1) / dimension))
    generator = np.random.default_rng(seed)
    faces = []
    for axis in range(dimension):
        for side in (0.0, 1.0):
            tangential = (
                qmc.LatinHypercube(d=dimension - 1, seed=generator).random(per_face)
                if dimension > 1
                else np.zeros((per_face, 0))
            )
            faces.append(np.insert(tangential, axis, side, axis=1))
    boundary_points = np.concatenate(faces)
    count = boundary_points.shape[0]
    if capacity - count < count:
        raise ValueError(
            f"Capacity {capacity} leaves fewer interior than boundary points."
        )
    points = np.concatenate(
        (boundary_points, cloud_points(capacity - count, dimension, seed))
    )
    return points, np.arange(capacity) < count


def measure_capacity(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    from phydrax.discretization import PointCloudPlan, PointDiffusionOperator
    from phydrax.discretization.meshfree import (
        LocalStencilPolicy,
        meshfree_multigrid_builder,
        MeshfreeCoarseningPolicy,
        MeshfreeHierarchyPlan,
    )
    from phydrax.linalg import (
        AbstractPreconditionerBuilder,
        ArraySpace,
        DiagonalLinearOperator,
        FailurePolicy,
        GaussSeidelPreconditionerBuilder,
        GMRES,
        ILUPreconditionerBuilder,
        LinearDerivativeSolvePolicy,
        LinearSolvePolicy,
        LinearSystem,
        plan_sparse_assembly,
        PreconditioningPolicy,
        prepare,
        prepare_sparse_assembly,
        refresh,
        refresh_sparse_assembly,
        SmoothedAggregationHierarchyBuilder,
        SmoothedAggregationPolicy,
        solve,
        SparseAssemblyPolicy,
        TolerancePolicy,
    )

    basis = comb(config.dimension + _STENCIL_DEGREE, _STENCIL_DEGREE)
    neighbors = min(capacity, 2 * basis)
    reservation = config.check_capacity(
        capacity, degree=_STENCIL_DEGREE, neighbors=neighbors
    )
    shared = PhaseRecorder()
    points, boundary = dirichlet_cube_cloud(capacity, config.dimension, seed)
    normals = points - np.mean(points, axis=0)
    cloud = shared.run(
        "local-fit",
        lambda: PointCloudPlan(
            points,
            np.full(capacity, 1 / capacity),
            boundary_mask=boundary,
            boundary_normals=normals,
            neighbors=neighbors,
            target_chunk_size=config.chunk_rows,
            stencil=LocalStencilPolicy(
                approximation="phs-rbf-fd",
                polynomial_degree=_STENCIL_DEGREE,
                chunk_rows=config.chunk_rows,
            ),
        ).prepare(),
        scope="PointCloudPlan.prepare (neighbor search and stencil fit fused)",
    )
    shared.unavailable(
        "search", "PointCloudPlan exposes combined neighborhood/stencil preparation"
    )
    diffusion = PointDiffusionOperator(cloud, 1, form="dissipative")
    stiffness = diffusion.stiffness()
    free = DiagonalLinearOperator(
        jnp.asarray(~boundary, dtype=points.dtype), space=stiffness.source
    )
    fixed = DiagonalLinearOperator(
        jnp.asarray(boundary, dtype=points.dtype), space=stiffness.source
    )
    fine = free @ stiffness @ free + fixed
    assembly_policy = SparseAssemblyPolicy(max_workspace_bytes=config.working_set_bytes)
    assembled = shared.run(
        "assembly",
        lambda: prepare_sparse_assembly(
            plan_sparse_assembly(fine, assembly_policy), fine
        ),
    )
    operator = assembled.operator
    storage = operator.sparse_storage()
    matrix = csr_matrix(
        (
            np.asarray(storage.values),
            np.asarray(storage.indices),
            np.asarray(storage.indptr),
        ),
        shape=(capacity, capacity),
    )
    expected = np.sin(np.pi * points[:, 0]) * np.prod(
        np.sin(np.pi * points[:, 1:]), axis=1
    )
    expected[boundary] = 0
    rhs = jnp.asarray(matrix @ expected)
    source_space = operator.source
    if not isinstance(source_space, ArraySpace):
        raise TypeError("Meshfree multilevel preparation requires an ArraySpace source.")
    hierarchy = shared.run(
        "hierarchy",
        lambda: MeshfreeHierarchyPlan(
            points,
            boundary=boundary,
            eliminated=boundary,
            policy=MeshfreeCoarseningPolicy(
                maximum_points=config.max_points,
                maximum_transfer_entries=config.resource_bytes // 32,
                chunk_rows=config.chunk_rows,
            ),
        ).prepare(source_space),
        scope="geometric-transfer-preparation",
    )
    sources = (
        ("native-ilu", ILUPreconditionerBuilder()),
        (
            "native-smoothed-aggregation",
            SmoothedAggregationHierarchyBuilder(
                SmoothedAggregationPolicy(
                    maximum_level_storage_bytes=config.working_set_bytes
                ),
                # Undamped Jacobi diverges here: rho(D^-1 A) is 3.4-5.5 on this
                # non-M-matrix stiffness (about half its off-diagonals are
                # positive). Symmetric Gauss-Seidel converges for any SPD level.
                GaussSeidelPreconditionerBuilder(),
                ILUPreconditionerBuilder(),
            ),
        ),
        (
            "native-meshfree-multilevel",
            meshfree_multigrid_builder(
                hierarchy, coarse_solver=ILUPreconditionerBuilder()
            ),
        ),
    )

    def measure_provider(
        provider: str, builder: AbstractPreconditionerBuilder
    ) -> dict[str, Any]:
        recorder = PhaseRecorder()
        policy = LinearSolvePolicy(
            GMRES(restart=min(40, capacity)),
            tolerance=TolerancePolicy(relative=1e-8, absolute=1e-10, max_steps=1000),
            derivative_solve=LinearDerivativeSolvePolicy(maximum_steps=1000),
            failure=FailurePolicy("error"),
            preconditioning=PreconditioningPolicy(builder, refresh="numeric"),
        )
        prepared = recorder.run(
            "hierarchy",
            lambda: prepare(LinearSystem(operator), policy),
            scope="native-solver-preconditioner-setup",
        )
        result, execution = recorder.compiled_action(
            lambda source: solve(prepared, source).value,
            rhs,
            budget_bytes=config.resource_bytes,
            repeats=config.repeats,
            scope=provider,
        )
        actual = np.asarray(result)
        full_result = recorder.run(
            "solve", lambda: solve(prepared, rhs), scope="diagnostic eager solve"
        )
        relative_residual = float(
            np.linalg.norm(matrix @ actual - np.asarray(rhs))
            / np.linalg.norm(np.asarray(rhs))
        )
        if not bool(np.asarray(full_result.successful)) or relative_residual > 1e-5:
            raise AssertionError(
                f"{provider} failed independent fine-system residual: {relative_residual}"
            )
        refreshed_diffusion = PointDiffusionOperator(cloud, 1.1, form="dissipative")
        refreshed_fine = free @ refreshed_diffusion.stiffness() @ free + fixed
        refreshed_assembly = recorder.run(
            "assembly",
            lambda: refresh_sparse_assembly(assembled, refreshed_fine),
            scope="numeric assembly refresh",
        )
        updated = recorder.run(
            "numeric-refresh",
            lambda: refresh(prepared, LinearSystem(refreshed_assembly.operator)),
        )
        refreshed_storage = refreshed_assembly.operator.sparse_storage()
        refreshed_matrix = csr_matrix(
            (
                np.asarray(refreshed_storage.values),
                np.asarray(refreshed_storage.indices),
                np.asarray(refreshed_storage.indptr),
            ),
            shape=(capacity, capacity),
        )
        refreshed_rhs = jnp.asarray(refreshed_matrix @ expected)
        refreshed_result = recorder.run(
            "solve", lambda: solve(updated, refreshed_rhs), scope="refreshed solve"
        )
        refreshed_residual = float(
            np.linalg.norm(
                refreshed_matrix @ np.asarray(refreshed_result.value)
                - np.asarray(refreshed_rhs)
            )
            / np.linalg.norm(np.asarray(refreshed_rhs))
        )
        if not bool(np.asarray(refreshed_result.successful)) or refreshed_residual > 1e-5:
            raise AssertionError(
                f"{provider} numeric refresh failed independent fine-system residual."
            )
        retained = logical_array_bytes((cloud, assembled, hierarchy, prepared, updated))
        if retained > min(config.working_set_bytes, config.resource_bytes):
            raise DeclaredCapacityRefusal(
                f"{provider} retained arrays exceed requested working-set/resource budget."
            )
        return {
            "provider": provider,
            "status": "measured",
            "phases": recorder.record(),
            **execution,
            "relative_residual": relative_residual,
            "solution_error": float(np.max(np.abs(actual - expected))),
            "iterations": int(np.asarray(full_result.diagnostics.iterations)),
            "matvec_count": int(np.asarray(full_result.diagnostics.matvec_count)),
            "independent_derivative_solve_policy": {
                "route": "krylov",
                "maximum_steps": 1000,
                "relative_tolerance": 1e-8,
                "absolute_tolerance": 1e-10,
            },
            "retained_bytes": retained,
            "refreshed_relative_residual": refreshed_residual,
            "refreshed_iterations": int(
                np.asarray(refreshed_result.diagnostics.iterations)
            ),
        }

    solvers: list[dict[str, Any]] = []
    for provider, builder in sources:
        try:
            solvers.append(measure_provider(provider, builder))
        except DeclaredCapacityRefusal:
            raise
        except Exception as error:  # One provider failure is its own row evidence.
            solvers.append(
                {
                    "provider": provider,
                    "status": "failed",
                    "failure": failure_record(error),
                }
            )
    evidence = hierarchy.evidence
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "domain": "unit-cube-convex-hull-dirichlet",
        "stencil": {
            "approximation": "phs-rbf-fd",
            "polynomial_degree": _STENCIL_DEGREE,
            "neighbors": neighbors,
            "form": "dissipative",
        },
        "boundary_points": int(np.count_nonzero(boundary)),
        "status": "measured",
        "phases": shared.record(),
        "solvers": solvers,
        "fine_nnz": matrix.nnz,
        "level_sizes": evidence.level_sizes,
        "reproduction_degree": evidence.reproduction_degree,
        "reproduction_residuals": evidence.reproduction_residuals,
        "near_nullspace_defects": evidence.near_nullspace_defects,
        "coarsening_rounds": evidence.coarsening_rounds,
        "grid_complexity": evidence.grid_complexity,
        "transfer_entries": evidence.transfer_entries,
        "stopping_reason": evidence.stopping_reason,
        "restriction_coordinates": evidence.restriction_coordinates,
        "reserved_working_set_bytes": reservation,
        "retained_bytes": max(
            (
                entry["retained_bytes"]
                for entry in solvers
                if entry["status"] == "measured"
            ),
            default=None,
        ),
        "oracle": "independent-host-SciPy sparse matrix manufactured RHS and true residual; not a continuum convergence claim",
        "optional_provider": {
            "status": "not-requested",
            "reason": "All three comparisons use native public owners",
        },
    }


def run(config: MeshfreeConfig = MeshfreeConfig(), /) -> dict[str, Any]:
    configure_precision(config, supported=("float64",))
    return make_record(config, admitted_rows(measure_capacity, config), Path(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    args = parser.parse_args()
    record = run(config_from_arguments(args))
    apply_baseline(record, args.baseline)
    if args.output is not None:
        write_json_atomic(args.output, record)
    else:
        print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

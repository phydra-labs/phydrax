# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compare native ILU, smoothed aggregation, and meshfree hierarchy on one fine system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull

from benchmarks._io import write_json_atomic
from benchmarks._runtime import logical_array_bytes, measure_synchronized
from benchmarks.meshfree_scaling import (
    add_config_arguments,
    apply_baseline,
    cloud_points,
    config_from_arguments,
    execution_evidence,
    make_record,
    measured_phase,
    MeshfreeConfig,
    unavailable_phases,
)


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
        DiagonalLinearOperator,
        FailurePolicy,
        GMRES,
        ILUPreconditionerBuilder,
        JacobiPreconditionerBuilder,
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

    reservation = config.check_capacity(capacity)
    points = cloud_points(capacity, config.dimension, seed)
    hull = ConvexHull(points)
    boundary = np.zeros(capacity, dtype=np.bool_)
    boundary[hull.vertices] = True
    normals = points - np.mean(points, axis=0)
    cloud, stencil_seconds = measure_synchronized(
        lambda: PointCloudPlan(
            points,
            np.full(capacity, 1 / capacity),
            boundary_mask=boundary,
            boundary_normals=normals,
            neighbors=config.neighbors,
            target_chunk_size=config.chunk_rows,
            stencil=LocalStencilPolicy(
                polynomial_degree=config.degree, chunk_rows=config.chunk_rows
            ),
        ).prepare()
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
    assembled, assembly_seconds = measure_synchronized(
        lambda: prepare_sparse_assembly(plan_sparse_assembly(fine, assembly_policy), fine)
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
    hierarchy, hierarchy_seconds = measure_synchronized(
        lambda: MeshfreeHierarchyPlan(
            points,
            boundary=boundary,
            policy=MeshfreeCoarseningPolicy(
                maximum_points=config.max_points,
                maximum_transfer_entries=config.resource_bytes // 32,
                chunk_rows=config.chunk_rows,
            ),
        ).prepare(operator.source)
    )
    sources = (
        ("native-ilu", ILUPreconditionerBuilder()),
        (
            "native-smoothed-aggregation",
            SmoothedAggregationHierarchyBuilder(
                SmoothedAggregationPolicy(
                    maximum_level_storage_bytes=config.working_set_bytes
                ),
                JacobiPreconditionerBuilder(),
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
    solvers: list[dict[str, Any]] = []
    for provider, builder in sources:
        policy = LinearSolvePolicy(
            GMRES(restart=min(40, capacity)),
            tolerance=TolerancePolicy(relative=1e-8, absolute=1e-10, max_steps=1000),
            derivative_solve=LinearDerivativeSolvePolicy(maximum_steps=1000),
            failure=FailurePolicy("error"),
            preconditioning=PreconditioningPolicy(builder, refresh="numeric"),
        )
        prepared, setup_seconds = measure_synchronized(
            lambda: prepare(LinearSystem(operator), policy)
        )
        execution = execution_evidence(
            lambda source: solve(prepared, source).value, rhs, config
        )
        actual = np.asarray(execution.pop("result"))
        full_result, diagnostic_seconds = measure_synchronized(
            lambda: solve(prepared, rhs)
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
        refreshed_assembly, numeric_assembly_seconds = measure_synchronized(
            lambda: refresh_sparse_assembly(assembled, refreshed_fine)
        )
        updated, refresh_seconds = measure_synchronized(
            lambda: refresh(prepared, LinearSystem(refreshed_assembly.operator))
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
        refreshed_result, refreshed_solve_seconds = measure_synchronized(
            lambda: solve(updated, refreshed_rhs)
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
        phases = unavailable_phases()
        phases.update(execution.pop("phases"))
        phases["hierarchy"] = {
            **measured_phase(setup_seconds),
            "scope": "native-solver-preconditioner-setup",
        }
        phases["numeric-refresh"] = {
            **measured_phase(refresh_seconds),
            "assembly_seconds": numeric_assembly_seconds,
        }
        retained = logical_array_bytes((cloud, assembled, hierarchy, prepared, updated))
        if retained > min(config.working_set_bytes, config.resource_bytes):
            raise ValueError(
                f"{provider} retained arrays exceed requested working-set/resource budget."
            )
        solvers.append(
            {
                "provider": provider,
                "status": "measured",
                "phases": phases,
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
                "diagnostic_execution_seconds": diagnostic_seconds,
                "retained_bytes": retained,
                "refreshed_solve_seconds": refreshed_solve_seconds,
                "refreshed_relative_residual": refreshed_residual,
                "refreshed_iterations": int(
                    np.asarray(refreshed_result.diagnostics.iterations)
                ),
            }
        )
    phases = unavailable_phases()
    phases["stencil"] = {
        **measured_phase(stencil_seconds),
        "includes_neighbor_preparation": True,
    }
    phases["neighbor"] = {
        "status": "unavailable",
        "reason": "PointCloudPlan exposes combined neighborhood/stencil preparation",
    }
    phases["assembly"] = measured_phase(assembly_seconds)
    phases["hierarchy"] = {
        **measured_phase(hierarchy_seconds),
        "scope": "geometric-transfer-preparation",
    }
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "domain": "unit-cube-convex-hull-dirichlet",
        "phases": phases,
        "solvers": solvers,
        "fine_nnz": matrix.nnz,
        "level_sizes": hierarchy.evidence.level_sizes,
        "constant_reproduction_residuals": hierarchy.evidence.constant_residuals,
        "linear_reproduction_residuals": hierarchy.evidence.linear_residuals,
        "transfer_entries": hierarchy.evidence.transfer_entries,
        "stopping_reason": hierarchy.evidence.stopping_reason,
        "reserved_working_set_bytes": reservation,
        "retained_bytes": max(entry["retained_bytes"] for entry in solvers),
        "oracle": "independent-host-SciPy sparse matrix manufactured RHS and true residual; not a continuum convergence claim",
        "optional_provider": {
            "status": "not-requested",
            "reason": "All three comparisons use native public owners",
        },
    }


def run(config: MeshfreeConfig = MeshfreeConfig(), /) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", config.precision == "float64")
    return make_record(
        config,
        [
            measure_capacity(size, seed, config)
            for size in config.sizes
            for seed in config.seeds
        ],
        Path(__file__),
    )


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

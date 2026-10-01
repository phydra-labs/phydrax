# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Run selectable Q1–Q8 numerical campaigns; evidence is not release.

Generic stencil controls govern bulk operators. Q2 has explicit elliptic
stencil, degree, and neighbor controls, independent of the generic policy.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from math import comb
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    benchmark_driver_fingerprint,
    capture_benchmark_identity,
    capture_environment,
    logical_array_bytes,
)
from benchmarks.meshfree_scaling import (
    add_config_arguments,
    cloud_points,
    config_from_arguments,
    MeshfreeConfig,
    PROJECT_ROOT,
)
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization import PointBoundaryKind, PointDiffusionForm
from phydrax.discretization.meshfree import LocalStencilPolicy, MeshfreeApproximation
from phydrax.linalg import SparseAssemblyPolicy


CAMPAIGNS = {
    "Q1": "polynomial-derivatives-and-row-refusal",
    "Q2": "boundary-elliptic-error",
    "Q3": "fine-system-preconditioner-comparison",
    "Q4": "conservative-exterior-and-coordinate-derivatives",
    "Q5": "sphere-torus-laplace-beltrami",
    "Q6": "expanding-surface-and-repeated-remap",
    "Q7": "langmuir-exchange-and-moving-window",
    "Q8": "learned-law-and-implicit-derivatives",
}


def gate(
    name: str,
    passed: bool | None,
    metrics: dict[str, Any],
    /,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Keep measured failures and missing measurements distinct."""
    return {
        "gate": name,
        "status": "unexecuted"
        if passed is None
        else "measured-passed"
        if passed
        else "failed",
        "metrics": metrics,
        "reason": reason,
    }


def upper_gate(
    name: str, metrics: dict[str, Any], key: str, tolerance: float, /
) -> dict[str, Any]:
    value = metrics.get(key)
    if value is None:
        return gate(name, None, {}, reason=f"Consumer did not measure {key}")
    if isinstance(value, bool):
        raise TypeError(f"{key} must be a numerical error, not a Boolean.")
    numeric = float(value)
    return gate(
        name,
        np.isfinite(numeric) and abs(numeric) <= tolerance,
        {key: numeric, "absolute_tolerance": tolerance},
    )


def boolean_gate(name: str, metrics: dict[str, Any], key: str, /) -> dict[str, Any]:
    value = metrics.get(key)
    if value is not None and type(value) is not bool:
        raise TypeError(f"{key} must be a measured Boolean or unavailable.")
    return gate(
        name,
        None if value is None else bool(value),
        {} if value is None else {key: bool(value)},
        reason=f"Consumer did not measure {key}" if value is None else None,
    )


def _q1(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    from phydrax.discretization.meshfree import (
        LocalStencilPolicy,
        MeshfreeFunctional,
        MeshfreeNeighborhoodPlan,
        prepare_local_stencils,
    )

    points = cloud_points(capacity, config.dimension, seed)
    neighbors = MeshfreeNeighborhoodPlan(
        points, config.neighbors, target_chunk_size=config.chunk_rows
    ).prepare()
    gradient_indices = tuple(
        tuple(int(d == axis) for d in range(config.dimension))
        for axis in range(config.dimension)
    )
    laplace_indices = tuple(
        tuple(2 * int(d == axis) for d in range(config.dimension))
        for axis in range(config.dimension)
    )
    functionals = tuple(
        MeshfreeFunctional((index,), np.asarray([1.0]), name=f"partial-{axis}")
        for axis, index in enumerate(gradient_indices)
    )
    functionals += (
        MeshfreeFunctional(laplace_indices, np.ones(config.dimension), name="laplacian"),
    )
    polynomial = np.sum(points**2, axis=1) + points[:, 0]
    smooth = np.exp(np.sum(points, axis=1))
    rows: list[dict[str, Any]] = []
    tolerance = 5e-3 if config.precision == "float32" else 1e-7
    for method in ("gmls", "phs-rbf-fd"):
        prepared = prepare_local_stencils(
            neighbors,
            points,
            points,
            functionals,
            LocalStencilPolicy(
                approximation=method,
                polynomial_degree=config.degree,
                chunk_rows=config.chunk_rows,
            ),
        )
        relation = prepared.neighborhood.relation

        def apply(values: np.ndarray, /) -> np.ndarray:
            return np.stack(
                [
                    np.asarray(
                        jnp.sum(
                            jnp.where(
                                relation.valid,
                                weights * jnp.asarray(values)[relation.source_indices],
                                0,
                            ),
                            axis=1,
                        )
                    )
                    for weights in prepared.weights
                ],
                axis=1,
            )

        expected = np.column_stack(
            (
                2 * points + np.eye(config.dimension)[0],
                np.full(capacity, 2 * config.dimension),
            )
        )
        actual = apply(polynomial)
        smooth_expected = np.column_stack(
            (np.broadcast_to(smooth[:, None], points.shape), config.dimension * smooth)
        )
        defects = np.sqrt(np.mean((apply(smooth) - smooth_expected) ** 2, axis=0))
        error = float(np.max(np.abs(actual - expected)))
        rows.append(
            {
                "method": method,
                "polynomial_error": error,
                "smooth_derivative_rms": defects.tolist(),
                "refused_rows": prepared.report.refused_rows,
                "maximum_condition_number": prepared.report.maximum_condition_number,
                "moment_residual": prepared.report.maximum_moment_residual,
                "retained_bytes": logical_array_bytes(prepared),
                "prepared_id": prepared.prepared_id,
                "gates": [
                    gate(
                        "analytic",
                        error <= tolerance,
                        {"error": error, "tolerance": tolerance},
                    ),
                    gate(
                        "row-acceptance",
                        prepared.report.refused_rows == 0,
                        {"refused_rows": prepared.report.refused_rows},
                    ),
                ],
            }
        )
    # A collapsed cloud is intentionally outside polynomial unisolvency. Only
    # the documented row-refusal exception is accepted, not arbitrary failures.
    line = np.zeros_like(points)
    line[:, 0] = np.linspace(0, 1, capacity)
    bad_neighbors = MeshfreeNeighborhoodPlan(
        line, config.neighbors, target_chunk_size=config.chunk_rows
    ).prepare()
    refused = prepare_local_stencils(
        bad_neighbors,
        line,
        line,
        functionals,
        LocalStencilPolicy(
            polynomial_degree=config.degree,
            chunk_rows=config.chunk_rows,
            acceptance="mask",
        ),
    )
    statuses, counts = np.unique(np.asarray(refused.evidence.status), return_counts=True)
    refusal = False
    try:
        prepare_local_stencils(
            bad_neighbors,
            line,
            line,
            functionals,
            LocalStencilPolicy(
                polynomial_degree=config.degree, chunk_rows=config.chunk_rows
            ),
        )
    except ValueError as error:
        if not str(error).startswith(
            "Meshfree stencil row "
        ) or "RANK_DEFICIENT" not in str(error):
            raise
        refusal = True
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "domain": "unit-cube-stratified",
        "methods": rows,
        "row_refusal": refusal,
        "degenerate_refused_rows": refused.report.refused_rows,
        "degenerate_row_status_counts": {
            str(int(status)): int(count)
            for status, count in zip(statuses, counts, strict=True)
        },
        "gates": [
            gate(
                "row-refusal",
                refusal and refused.report.refused_rows == capacity,
                {
                    "degenerate-cloud": "collinear",
                    "refused_rows": refused.report.refused_rows,
                },
            )
        ],
        "oracle": "independent-host-NumPy polynomial and exponential derivatives",
    }


def _consumer(
    module: str,
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    /,
    *,
    surface: bool = False,
    learned: bool = False,
    options: dict[str, str | int | LocalStencilPolicy | SparseAssemblyPolicy]
    | None = None,
) -> dict[str, Any]:
    if surface and config.dimension != 3:
        raise ValueError(
            f"{module} requires ambient dimension=3; requested dimension is not silently changed."
        )
    consumer_module = importlib.import_module(module)
    module_file = consumer_module.__file__
    if module_file is None:
        raise TypeError(
            f"{module} must expose a concrete source file for qualification provenance."
        )
    function: Callable[..., dict[str, Any]] = getattr(consumer_module, "run_workflow")
    arguments: dict[str, Any] = {
        "size": capacity,
        "dimension": config.dimension,
        "seed": seed,
    }
    if options is not None:
        arguments.update(options)
    if learned:
        arguments["steps"] = config.steps
    metrics = function(**arguments)
    if not isinstance(metrics, dict):
        raise TypeError(f"{module}.run_workflow must return a numerical record.")
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": arguments["dimension"],
        "consumer": f"{module}.run_workflow",
        "metrics": metrics,
        "consumer_source_fingerprint": benchmark_driver_fingerprint(
            PROJECT_ROOT, Path(module_file)
        ),
        "oracle_provenance": metrics.get("oracle_provenance"),
        "domain": metrics.get("domain", metrics.get("training_domain")),
    }


def _q2(
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    /,
    *,
    boundary_kind: PointBoundaryKind = "dirichlet",
    form: PointDiffusionForm = "collocated",
    stencil: LocalStencilPolicy | None = None,
    neighbors: int | None = None,
) -> dict[str, Any]:
    policy = (
        LocalStencilPolicy(
            approximation="phs-rbf-fd", polynomial_degree=3, chunk_rows=config.chunk_rows
        )
        if stencil is None
        else stencil
    )
    basis = comb(config.dimension + policy.polynomial_degree, policy.polynomial_degree)
    neighbor_capacity = min(capacity, 2 * basis) if neighbors is None else neighbors
    stencil_reservation = config.check_capacity(
        capacity, degree=policy.polynomial_degree, neighbors=neighbor_capacity
    )
    assembly_reservation = (
        64
        * capacity
        * neighbor_capacity
        * (
            config.dimension * neighbor_capacity
            if form == "dissipative"
            else 2 * config.dimension + 4
        )
    )
    reservation = stencil_reservation + assembly_reservation
    if reservation > min(config.working_set_bytes, config.resource_bytes):
        raise ValueError(
            "Selected elliptic stencil/saddle/assembly workload exceeds requested budgets."
        )
    assembly_policy = SparseAssemblyPolicy(
        max_workspace_bytes=min(config.working_set_bytes, config.resource_bytes)
        - stencil_reservation
    )
    options: dict[str, str | int | LocalStencilPolicy | SparseAssemblyPolicy] = {
        "manufactured": "exponential",
        "boundary_kind": boundary_kind,
        "form": form,
        "stencil": policy,
        "target_chunk_size": config.chunk_rows,
        "assembly_policy": assembly_policy,
    }
    if neighbors is not None:
        options["neighbors"] = neighbors
    row = _consumer(
        "examples.point_cloud_poisson", capacity, seed, config, options=options
    )
    metrics = row["metrics"]
    row["boundary_kind"] = boundary_kind
    row["form"] = form
    row["reserved_working_set_bytes"] = reservation
    row["elliptic_policy"] = {
        "approximation": policy.approximation,
        "polynomial_degree": policy.polynomial_degree,
        "neighbor_capacity": neighbor_capacity,
        "neighbor_policy": "native-support-count" if neighbors is None else "explicit",
        "chunk_rows": policy.chunk_rows,
        "target_chunk_size": config.chunk_rows,
        "stencil_saddle_reservation_bytes": stencil_reservation,
        "assembly_reservation_bytes": assembly_reservation,
        "native_assembly_workspace_budget": assembly_policy.max_workspace_bytes,
    }
    row["qualification_scope"] = (
        "projected-source-point-gauged-Neumann"
        if boundary_kind == "neumann"
        else "independent-continuum-MMS"
    )
    row["gates"] = [
        upper_gate("analytic", metrics, "maximum_solution_error", 5e-2),
        upper_gate("boundary-equations", metrics, "boundary_residual_norm", 1e-5),
        upper_gate("boundary-field-error", metrics, "maximum_boundary_error", 5e-2),
        upper_gate("true-residual", metrics, "residual_norm", 1e-5),
        boolean_gate("solve-status", metrics, "successful"),
    ]
    if boundary_kind == "dirichlet":
        row["gates"].append(
            upper_gate("exact-Dirichlet-trace", metrics, "maximum_boundary_error", 1e-5)
        )
    if form == "dissipative":
        row["gates"].append(boolean_gate("configured-sbp", metrics, "sbp_passed"))
    return row


def _q3(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    from benchmarks.meshfree_multilevel import measure_capacity

    row = measure_capacity(capacity, seed, config)
    measured_costs = all(
        entry["phases"]["hierarchy"]["status"] == "measured"
        and np.isfinite(entry["phases"]["hierarchy"]["seconds"])
        and entry["phases"]["hierarchy"]["seconds"] >= 0
        and entry["iterations"] >= 0
        for entry in row["solvers"]
    )
    row["gates"] = [
        gate(
            "fine-system-cost",
            measured_costs,
            {"providers": [entry["provider"] for entry in row["solvers"]]},
        ),
        gate(
            "true-residual",
            all(entry["relative_residual"] <= 1e-5 for entry in row["solvers"]),
            {
                "residuals": {
                    entry["provider"]: entry["relative_residual"]
                    for entry in row["solvers"]
                }
            },
        ),
        gate(
            "transfer-reproduction",
            all(
                abs(value) <= 1e-7
                for value in (
                    *row["constant_reproduction_residuals"],
                    *row["linear_reproduction_residuals"],
                )
            ),
            {
                "constant": row["constant_reproduction_residuals"],
                "linear": row["linear_reproduction_residuals"],
            },
        ),
        gate(
            "analytic",
            all(entry["solution_error"] <= 1e-5 for entry in row["solvers"]),
            {
                "errors": {
                    entry["provider"]: entry["solution_error"] for entry in row["solvers"]
                }
            },
        ),
    ]
    return row


def _q4(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    row = _consumer("examples.meshfree_conservative_diffusion", capacity, seed, config)
    metrics = row["metrics"]
    row["gates"] = [
        upper_gate("moment", metrics, "moment_residual", 1e-6),
        upper_gate("analytic", metrics, "quadratic_laplacian_error", 1e-6),
        upper_gate("conservation", metrics, "conservation_residual", 1e-7),
        upper_gate(
            "advection-conservation", metrics, "advection_conservation_residual", 1e-7
        ),
        boolean_gate("coercivity", metrics, "reduced_spd"),
        boolean_gate("feasibility", metrics, "metric_accepted"),
        boolean_gate("nonnegative-refusal", metrics, "nonnegative_refused"),
        gate(
            "nonnegative-witness",
            None
            if metrics.get("negative_weights") is None
            or metrics.get("metric_accepted") is None
            else metrics["negative_weights"] == 0 and metrics["metric_accepted"],
            {
                "negative_weights": metrics.get("negative_weights"),
                "metric_status": metrics.get("metric_status"),
            },
        ),
        boolean_gate(
            "coordinate-derivative-availability", metrics, "derivative_available"
        ),
        upper_gate("coordinate-derivative", metrics, "derivative_error", 1e-3),
    ]
    return row


def _q5(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    reservation = 4 * config.check_capacity(capacity)
    if reservation > min(config.working_set_bytes, config.resource_bytes):
        raise ValueError(
            "Simultaneous exact/sampled sphere/torus reservations exceed requested budgets."
        )
    row = _consumer(
        "examples.meshfree_surface_laplace_beltrami", capacity, seed, config, surface=True
    )
    metrics = row["metrics"]
    row["reserved_working_set_bytes"] = reservation
    row["reservation_scope"] = "four simultaneous exact/sampled sphere/torus preparations"
    row["gates"] = [
        upper_gate("sphere-analytic", metrics, "sphere_exact_error", 1.0),
        upper_gate("sphere-sampled", metrics, "sphere_sampled_error", 1.0),
        upper_gate("torus-analytic", metrics, "torus_exact_error", 1.0),
        upper_gate("torus-sampled", metrics, "torus_sampled_error", 1.0),
        upper_gate("sphere-normal-comparison", metrics, "sphere_normal_error", 0.3),
        upper_gate("torus-normal-comparison", metrics, "torus_normal_error", 0.3),
    ]
    return row


def _q6(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    row = _consumer(
        "examples.meshfree_moving_surface_reaction_diffusion",
        capacity,
        seed,
        config,
        surface=True,
    )
    metrics = row["metrics"]
    row["gates"] = [
        upper_gate("dilution", metrics, "concentration_error", 5e-3),
        upper_gate("conservation", metrics, "conservation_residual", 1e-6),
        upper_gate("repeated-shift", metrics, "shift_conservation_residual", 1e-6),
        upper_gate("epoch-remap", metrics, "epoch_conservation_residual", 1e-6),
        upper_gate(
            "repeated-remap", metrics, "repeated_cycle_conservation_residual", 1e-6
        ),
        upper_gate("rollback", metrics, "rollback_error", 1e-8),
        boolean_gate("step-status", metrics, "successful"),
    ]
    return row


def _q7(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    row = _consumer(
        "examples.meshfree_bulk_surface_exchange", capacity, seed, config, surface=True
    )
    metrics = row["metrics"]
    row["gates"] = [
        upper_gate("analytic", metrics, "equilibrium_error", 1e-4),
        upper_gate("conservation", metrics, "conservation_error", 1e-6),
        upper_gate("window-lag", metrics, "window_lag_error", 1e-4),
        boolean_gate("window-status", metrics, "accepted"),
    ]
    return row


def _q8(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    if config.precision != "float64":
        raise ValueError(
            "Learned implicit recovery requires float64; requested precision is not silently changed."
        )
    unseen_reservation = config.check_capacity(capacity + 2 * config.dimension)
    reservation = config.check_capacity(capacity) + unseen_reservation
    if reservation > min(config.working_set_bytes, config.resource_bytes):
        raise ValueError(
            "Simultaneous training and unseen-cloud reservations exceed requested budgets."
        )
    row = _consumer(
        "examples.meshfree_learned_edge_flux", capacity, seed, config, learned=True
    )
    metrics = row["metrics"]
    row["reserved_working_set_bytes"] = reservation
    row["gates"] = [
        upper_gate("law-recovery", metrics, "parameter_error", 5e-2),
        upper_gate("implicit-derivative", metrics, "gradient_relative_error", 1e-3),
        boolean_gate("coverage", metrics, "coverage_refused"),
        boolean_gate("newton-refusal", metrics, "failure_rejected"),
        boolean_gate("primal-status", metrics, "primal_successful"),
        boolean_gate("adjoint-status", metrics, "adjoint_successful"),
        upper_gate("conservation", metrics, "conservation_defect", 1e-7),
    ]
    return row


RUNNERS = {
    "Q1": _q1,
    "Q2": _q2,
    "Q3": _q3,
    "Q4": _q4,
    "Q5": _q5,
    "Q6": _q6,
    "Q7": _q7,
    "Q8": _q8,
}


def _convergence(
    rows: list[dict[str, Any]], campaign: str, config: MeshfreeConfig, /
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    scopes: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        scope = (row.get("boundary_kind", "native"), row.get("form", "native"))
        scopes.setdefault(scope, []).append(row)
    for scope, scoped_rows in sorted(scopes.items()):
        for seed in config.seeds:
            selected = sorted(
                (row for row in scoped_rows if row["seed"] == seed),
                key=lambda row: row["capacity"],
            )
            coordinates = {"seed": seed, "boundary_kind": scope[0], "form": scope[1]}
            if len(selected) < 2:
                gates.append(
                    gate(
                        "convergence",
                        None,
                        coordinates,
                        reason="At least two requested capacities are required",
                    )
                )
                continue
            if campaign == "Q1":
                for method in ("gmls", "phs-rbf-fd"):
                    errors = [
                        next(item for item in row["methods"] if item["method"] == method)[
                            "smooth_derivative_rms"
                        ]
                        for row in selected
                    ]
                    h_first = selected[0]["capacity"] ** (-1 / config.dimension)
                    h_last = selected[-1]["capacity"] ** (-1 / config.dimension)
                    for axis in range(config.dimension + 1):
                        first, last = float(errors[0][axis]), float(errors[-1][axis])
                        rate = (
                            np.log(first / last) / np.log(h_first / h_last)
                            if first > 0 and last > 0 and h_first > h_last > 0
                            else None
                        )
                        gates.append(
                            gate(
                                "convergence",
                                None if rate is None else bool(rate > 0),
                                {
                                    **coordinates,
                                    "method": method,
                                    "derivative": axis,
                                    "errors": [values[axis] for values in errors],
                                    "observed_rate": rate,
                                    "refinement_scale": "nominal N^(-1/d)",
                                    "polynomial_degree": config.degree,
                                },
                            )
                        )
                continue
            if campaign == "Q2":
                keys = ["maximum_solution_error"]
                if scope[0] != "dirichlet":
                    keys.append("maximum_boundary_error")
                if scope[0] == "neumann":
                    keys.append("maximum_source_correction")
                scales = [row["metrics"].get("spacing") for row in selected]
                scale_provenance = "consumer nominal Cartesian axis spacing"
            elif campaign == "Q5":
                keys = [
                    "sphere_exact_error",
                    "sphere_sampled_error",
                    "torus_exact_error",
                    "torus_sampled_error",
                ]
                scales = [row["capacity"] ** (-0.5) for row in selected]
                scale_provenance = "nominal N^(-1/2) intrinsic surface density"
            else:
                continue
            for key in keys:
                errors = [row["metrics"].get(key) for row in selected]
                if any(value is None for value in (*errors, *scales)):
                    gates.append(
                        gate(
                            "convergence",
                            None,
                            {**coordinates, "metric": key},
                            reason="Missing independent error or refinement scale measurement",
                        )
                    )
                    continue
                first, last = float(errors[0]), float(errors[-1])
                h_first, h_last = float(scales[0]), float(scales[-1])
                rate = (
                    np.log(first / last) / np.log(h_first / h_last)
                    if first > 0 and last > 0 and h_first > h_last > 0
                    else None
                )
                gates.append(
                    gate(
                        "convergence",
                        None if rate is None else bool(rate > 0),
                        {
                            **coordinates,
                            "metric": key,
                            "errors": errors,
                            "scales": scales,
                            "scale_provenance": scale_provenance,
                            "observed_rate": rate,
                            "polynomial_degree": selected[0]["metrics"].get(
                                "polynomial_degree"
                            ),
                        },
                    )
                )
    return gates


def run(
    config: MeshfreeConfig = MeshfreeConfig(sizes=(256, 512), dimension=3),
    campaigns: tuple[str, ...] = tuple(CAMPAIGNS),
    /,
    *,
    elliptic_boundaries: tuple[PointBoundaryKind, ...] = (
        "dirichlet",
        "neumann",
        "robin",
    ),
    elliptic_forms: tuple[PointDiffusionForm, ...] = ("collocated",),
    elliptic_stencil: MeshfreeApproximation = "phs-rbf-fd",
    elliptic_degree: int = 3,
    elliptic_neighbors: int | None = None,
) -> dict[str, Any]:
    if (
        not campaigns
        or any(name not in CAMPAIGNS for name in campaigns)
        or len(set(campaigns)) != len(campaigns)
    ):
        raise ValueError("Select distinct Q1..Q8 campaign identifiers.")
    if (
        not elliptic_boundaries
        or len(set(elliptic_boundaries)) != len(elliptic_boundaries)
        or any(
            kind not in ("dirichlet", "neumann", "robin") for kind in elliptic_boundaries
        )
    ):
        raise ValueError("Select distinct native elliptic boundary kinds.")
    if (
        not elliptic_forms
        or len(set(elliptic_forms)) != len(elliptic_forms)
        or any(form not in ("collocated", "dissipative") for form in elliptic_forms)
    ):
        raise ValueError("Select distinct native elliptic forms.")
    elliptic_policy = LocalStencilPolicy(
        approximation=elliptic_stencil,
        polynomial_degree=elliptic_degree,
        chunk_rows=config.chunk_rows,
    )
    jax.config.update("jax_enable_x64", config.precision == "float64")
    records: list[dict[str, Any]] = []
    for name in campaigns:
        if name == "Q2":
            rows = [
                _q2(
                    size,
                    seed,
                    config,
                    boundary_kind=kind,
                    form=form,
                    stencil=elliptic_policy,
                    neighbors=elliptic_neighbors,
                )
                for kind in elliptic_boundaries
                for form in elliptic_forms
                for size in config.sizes
                for seed in config.seeds
            ]
        else:
            rows = [
                RUNNERS[name](size, seed, config)
                for size in config.sizes
                for seed in config.seeds
            ]
        for row in rows:
            row["reserved_working_set_bytes"] = max(
                row.get("reserved_working_set_bytes", 0),
                config.check_capacity(row["capacity"]),
            )
            row["gates"] = [
                *row.get("gates", []),
                gate(
                    "public-workflow",
                    True,
                    {"consumer": row.get("consumer", RUNNERS[name].__module__)},
                ),
            ]
            row["resource_reservation"] = {
                "status": "provisional",
                "reserved_bytes": row["reserved_working_set_bytes"],
                "working_set_budget": config.working_set_bytes,
                "resource_budget": config.resource_bytes,
                "reason": "Declared workload reservation, not measured peak memory",
            }
            retained = row.get(
                "retained_bytes", row.get("metrics", {}).get("retained_bytes")
            )
            if retained is None and name == "Q1":
                retained = max(method["retained_bytes"] for method in row["methods"])
            if retained is not None and int(retained) > config.working_set_bytes:
                raise ValueError(
                    f"{name} retained arrays exceed requested working-set budget."
                )
            row["gates"].append(
                gate(
                    "resource",
                    None
                    if retained is None
                    else int(retained) <= config.working_set_bytes,
                    {} if retained is None else {"retained_bytes": int(retained)},
                    reason="Consumer does not expose retained arrays; reservation is not measured peak memory"
                    if retained is None
                    else "Measured logical retained payload; not physical peak memory",
                )
            )
        convergence = _convergence(rows, name, config)
        measured = [item for row in rows for item in row.get("gates", [])] + convergence
        method_gates = [
            item
            for row in rows
            for method in row.get("methods", [])
            for item in method["gates"]
        ]
        measured += method_gates
        status = (
            "failed"
            if any(item["status"] == "failed" for item in measured)
            else "provisional"
            if any(item["status"] == "unexecuted" for item in measured)
            else "measured-passed"
        )
        records.append(
            {
                "campaign_id": name,
                "name": CAMPAIGNS[name],
                "status": "failed" if status == "failed" else "provisional",
                "numerical_status": status,
                "rows": rows,
                "convergence": convergence,
                "release_gates": [
                    gate(
                        "scaling",
                        None,
                        {},
                        reason="Requires separately executed phase benchmark and approved resource envelope",
                    ),
                    gate(
                        "resource-peak",
                        None,
                        {},
                        reason="Logical payload and compiler estimates are not a measured peak resident envelope",
                    ),
                ],
                "released": False,
            }
        )
    record: dict[str, Any] = {
        "kind": "meshfree-qualification-campaign",
        "config": asdict(config),
        "campaigns": records,
        "unexecuted_campaigns": [name for name in CAMPAIGNS if name not in campaigns],
        "environment": capture_environment().to_dict(),
        "precision": config.precision,
        "released": False,
    }
    record["generic_stencil_policy"] = {
        "applies_to": ["Q1", "Q3"],
        "polynomial_degree": config.degree,
        "neighbors": config.neighbors,
        "elliptic_policy": "separate elliptic_selection namespace",
    }
    record["elliptic_selection"] = {
        "status": "executed" if "Q2" in campaigns else "unexecuted",
        "manufactured": "exponential",
        "boundary_kinds": list(elliptic_boundaries),
        "forms": list(elliptic_forms),
        "stencil": elliptic_stencil,
        "polynomial_degree": elliptic_degree,
        "neighbors": elliptic_neighbors,
        "neighbor_policy": "native-support-count"
        if elliptic_neighbors is None
        else "explicit",
    }
    dependencies = (
        ("benchmarks/meshfree_scaling.py", "benchmarks/meshfree_multilevel.py")
        if "Q3" in campaigns
        else ("benchmarks/meshfree_scaling.py",)
    )
    record["driver_dependencies"] = {
        path: benchmark_driver_fingerprint(PROJECT_ROOT, PROJECT_ROOT / path)
        for path in dependencies
    }
    record["consumer_sources"] = {
        row["consumer"]: row["consumer_source_fingerprint"]
        for campaign in records
        for row in campaign["rows"]
        if "consumer" in row
    }
    record["identity"] = capture_benchmark_identity(
        PROJECT_ROOT, Path(__file__), tuple(record)
    ).to_dict()
    record["campaign_definition_id"] = canonical_fingerprint(
        {
            "selectors": list(campaigns),
            "config": asdict(config),
            "source": record["identity"],
            "dependencies": record["driver_dependencies"],
            "consumers": record["consumer_sources"],
            "elliptic_selection": record["elliptic_selection"],
        }
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.set_defaults(dimension=3, sizes=[256, 512])
    parser.add_argument(
        "--campaigns", nargs="+", choices=tuple(CAMPAIGNS), default=list(CAMPAIGNS)
    )
    parser.add_argument(
        "--elliptic-boundaries",
        nargs="+",
        choices=("dirichlet", "neumann", "robin"),
        default=["dirichlet", "neumann", "robin"],
    )
    parser.add_argument(
        "--elliptic-forms",
        nargs="+",
        choices=("collocated", "dissipative"),
        default=["collocated"],
    )
    parser.add_argument(
        "--elliptic-stencil", choices=("gmls", "phs-rbf-fd"), default="phs-rbf-fd"
    )
    parser.add_argument("--elliptic-degree", type=int, default=3)
    parser.add_argument(
        "--elliptic-neighbors",
        type=int,
        help="Explicit Q2 width; default is native support count",
    )
    args = parser.parse_args()
    if args.baseline is not None:
        parser.error(
            "Qualification does not perform before/after timing comparisons; use benchmark drivers."
        )
    record = run(
        config_from_arguments(args),
        tuple(args.campaigns),
        elliptic_boundaries=tuple(args.elliptic_boundaries),
        elliptic_forms=tuple(args.elliptic_forms),
        elliptic_stencil=args.elliptic_stencil,
        elliptic_degree=args.elliptic_degree,
        elliptic_neighbors=args.elliptic_neighbors,
    )
    if args.output is not None:
        write_json_atomic(args.output, record)
    else:
        print(json.dumps(record, indent=2, allow_nan=False))
    if any(campaign["status"] == "failed" for campaign in record["campaigns"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

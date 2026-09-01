#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_repeated,
    measure_synchronized,
)


def _affine_geometry(grid: phx.discretization.iga.BSplineGrid):
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    return phx.discretization.iga.NURBSGeometryState(
        jnp.stack((xx, yy), axis=-1),
        jnp.ones((grid.coefficient_count, grid.coefficient_count)),
    )


def _case(degree: int, span_count: int, *, warmup: int, repeats: int):
    grid = phx.discretization.iga.BSplineGrid.open_uniform(
        degree,
        span_count,
        interval=(0.0, 1.0),
    )
    geometry = _affine_geometry(grid)
    plan = phx.discretization.iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        field_name="u",
        axis_names=("xi", "eta"),
        quadrature_policy=phx.discretization.iga.IsogeometricQuadraturePolicy(degree + 1),
        qualification_policy=phx.discretization.iga.IsogeometricH1QualificationPolicy(),
    )
    discretization, preparation_seconds = measure_synchronized(
        lambda: plan.prepare(numeric_version=f"timing-p{degree}-n{span_count}")
    )
    constraint = discretization.homogeneous_trace_constraint("u")
    source = phx.equations.coefficient(
        lambda points, args: (
            2.0
            * jnp.pi**2
            * jnp.sin(jnp.pi * points[..., 0])
            * jnp.sin(jnp.pi * points[..., 1])
        ),
        coefficient_id="iga-h1-timing-source",
    )
    form = phx.equations.FiniteElementForm(
        f"iga-h1-timing-p{degree}-n{span_count}",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", source),
        ),
    )
    compiled, compilation_seconds = measure_synchronized(
        lambda: phx.equations.compile_finite_element_problem(
            form,
            discretization,
            constraint=constraint,
            execution_policy=phx.equations.FiniteElementExecutionPolicy(
                realization="matrix_free",
                local_kernel="sum_factorized",
            ),
        )
    )
    operator = compiled.affine_operator()
    probe = jnp.linspace(0.25, 1.25, compiled.state_space.size)
    _, first_apply_seconds = measure_synchronized(lambda: operator.mv(probe))
    _, apply_distribution = measure_repeated(
        lambda: operator.mv(probe),
        warmup=warmup,
        repeats=repeats,
    )
    system, right_hand_side = compiled.linear_system()
    result, first_solve_seconds = measure_synchronized(
        lambda: phx.linalg.solve(system, right_hand_side)
    )
    _, solve_distribution = measure_repeated(
        lambda: phx.linalg.solve(system, right_hand_side),
        warmup=warmup,
        repeats=repeats,
    )
    residual = compiled.residual(result.value)
    residual_norm = jnp.sqrt(jnp.real(jnp.vdot(residual, residual)))
    right_hand_side_norm = jnp.sqrt(jnp.real(jnp.vdot(right_hand_side, right_hand_side)))
    evidence = discretization.default_geometry_evidence
    return {
        "apply": apply_distribution.to_seconds_dict(),
        "coefficient_count": grid.coefficient_count**2,
        "compilation_seconds": compilation_seconds,
        "degree": degree,
        "first_apply_seconds": first_apply_seconds,
        "first_solve_seconds": first_solve_seconds,
        "free_coefficient_count": compiled.state_space.size,
        "geometry_evidence": {
            "minimum_orientation_ratio": float(evidence.minimum_orientation_ratio),
            "minimum_rank_ratio": float(evidence.minimum_rank_ratio),
            "minimum_weight_ratio": float(evidence.minimum_weight_ratio),
        },
        "logical_payload_bytes": logical_array_bytes((discretization, compiled)),
        "normalized_free_residual": float(
            residual_norm / jnp.maximum(right_hand_side_norm, 1.0)
        ),
        "preparation_seconds": preparation_seconds,
        "solve": solve_distribution.to_seconds_dict(),
        "solver_successful": bool(jnp.all(result.successful)),
        "span_count_per_axis": span_count,
    }


def run(*, smoke: bool, warmup: int, repeats: int):
    spans = (2, 4) if smoke else (4, 8, 16)
    rows = [
        _case(degree, span_count, warmup=warmup, repeats=repeats)
        for degree in (2, 3, 4)
        for span_count in spans
    ]
    return {
        "environment": capture_environment().to_dict(),
        "kind": "iga-h1-record-only-timing",
        "record_only": True,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record S1 matrix-free isogeometric H1 timings."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/iga_h1.json"),
    )
    arguments = parser.parse_args()
    payload = run(
        smoke=arguments.smoke,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    print(rendered)
    write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()

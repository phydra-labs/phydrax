#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Phase-separated campaign for analytic periodic seam enforcement.

Each case prepares one analytic periodic projection of a free MLP on the unit box
``[0, 1]^d`` periodic in its first ``axes`` coordinates, with every coordinate
derivative order ``0..max_order`` matched on every seam. The phases are host
preparation (endpoint right inverses and program compilation), JAX lowering and
XLA compilation of the projected-field evaluation, the first synchronized
execution, warm execution, and a compiled derivative evaluation (first
coordinate derivative plus the model-parameter gradient of a mean-square
objective). Seam residuals are checked before any timing. The free model at the
same query count is the reference; the projection changes the field, so the
comparison is execution cost only, not accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

import phydrax as phx

from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


_WIDTH = 32
_DEPTH = 2
_SEAM_PROBES = 64


def _unit_box(dimension: int, /) -> phx.domain.HyperRectangle:
    return phx.domain.HyperRectangle(
        jnp.zeros((dimension,), dtype=jnp.float64),
        jnp.ones((dimension,), dtype=jnp.float64),
    )


def _model(dimension: int, /) -> phx.nn.models.MLP:
    return phx.nn.models.MLP(
        in_size=dimension,
        out_size="scalar",
        width_size=_WIDTH,
        depth=_DEPTH,
        activation=jnp.tanh,
        key=jr.key(0),
    )


def _compiled(
    function: Callable[..., Any], *arguments: Any
) -> tuple[Any, dict[str, Any]]:
    executable, compilation = measure_lower_and_compile(
        lambda: jax.jit(function).lower(*arguments), lambda lowered: lowered.compile()
    )
    compiler = compiler_evidence(
        executable.cost_analysis(), executable.memory_analysis(), source="xla"
    )
    return executable, {"compilation": asdict(compilation), "compiler": asdict(compiler)}


def _timed(
    executable: Any, arguments: tuple[Any, ...], repeats: int, /
) -> dict[str, Any]:
    _, first = measure_synchronized(lambda: executable(*arguments))
    _, warm = measure_repeated(lambda: executable(*arguments), warmup=1, repeats=repeats)
    return {"first_execution_seconds": first, "warm": warm.to_dict(unit="seconds")}


def _case(axes: int, max_order: int, queries: int, repeats: int, /) -> dict[str, Any]:
    dimension = max(axes, 2)
    box = _unit_box(dimension)
    model = _model(dimension)
    parameters, static = eqx.partition(model, eqx.is_inexact_array)
    u = box.Model("x")(model)
    identifications = tuple(
        phx.domain.PeriodicIdentification(box, "x", component=axis)
        for axis in range(axes)
    )
    conditions = tuple(
        phx.conditions.Periodic("u", identification.pairing(), order=order)
        for identification in identifications
        for order in range(max_order + 1)
    )

    def prepare() -> tuple[
        phx.enforcement.PreparedPeriodicProjection, phx.enforcement.EnforcementProgram
    ]:
        prepared = phx.enforcement.prepare_periodic_projection(
            {"u": u}, conditions, route="analytic"
        )
        program = phx.enforcement.compile(
            {"u": u},
            (phx.enforcement.EnforcementSpec(prepared.condition, realization=prepared),),
        )
        return prepared, program

    (prepared, program), preparation = measure_synchronized(prepare)
    projected = program.apply({"u": u})["u"]
    seam_defect = max(
        float(
            jnp.max(
                jnp.abs(
                    condition.residual({"u": projected})(
                        condition.on.sample(
                            phx.domain.PointSampling(_SEAM_PROBES), key=jr.key(1)
                        )
                    ).data
                )
            )
        )
        for condition in conditions
    )
    if not seam_defect <= 1e-10:
        raise RuntimeError(f"Seam defect {seam_defect:.3e} before timing.")

    points = jr.uniform(jr.key(2), (queries, dimension), dtype=jnp.float64)
    component = box.component()

    def field(values: Any, enforced: bool, /) -> phx.domain.DomainFunction:
        free = box.Model("x")(eqx.combine(values, static))
        return program.apply({"u": free})["u"] if enforced else free

    def evaluate_projected(values: Any, coordinates: Array) -> Array:
        return jnp.asarray(field(values, True)(component.points({"x": coordinates})).data)

    def evaluate_free(values: Any, coordinates: Array) -> Array:
        return jnp.asarray(
            field(values, False)(component.points({"x": coordinates})).data
        )

    def derivative(values: Any, coordinates: Array) -> tuple[Array, Any]:
        batch = component.points({"x": coordinates})

        def objective(current: Any) -> Array:
            return jnp.mean(jnp.asarray(field(current, True)(batch).data) ** 2)

        slope = phx.operators.partial_n(field(values, True), var="x", axis=0, order=1)
        return jnp.asarray(slope(batch).data), eqx.filter_grad(objective)(values)

    phases: dict[str, Any] = {}
    for name, function in (
        ("free", evaluate_free),
        ("projected", evaluate_projected),
        ("derivative", derivative),
    ):
        executable, compiled = _compiled(function, parameters, points)
        phases[name] = {**compiled, **_timed(executable, (parameters, points), repeats)}
        print(
            f"axes={axes} max_order={max_order} queries={queries} {name}: "
            f"compile {compiled['compilation']['compilation_seconds']:.2f} s, "
            f"warm median {phases[name]['warm']['median_seconds']:.4f} s",
            file=sys.stderr,
            flush=True,
        )
    return {
        "axes": axes,
        "max_order": max_order,
        "declarations": len(conditions),
        "queries": queries,
        "dimension": dimension,
        "model": {"width": _WIDTH, "depth": _DEPTH},
        "status": "measured",
        "seam_defect": seam_defect,
        "endpoint_evaluations": prepared.evidence.endpoint_evaluations,
        "axis_evidence": [
            {
                "rows": axis.rows,
                "rank": axis.rank,
                "basis_size": axis.basis_size,
                "condition_number": axis.condition_number,
            }
            for axis in prepared.evidence.axes
        ],
        "preparation_seconds": preparation,
        "retained_array_bytes": logical_array_bytes(prepared),
        "phases": phases,
    }


def _refusal(axes: int, limit: int, /) -> dict[str, Any]:
    box = _unit_box(axes)
    u = box.Model("x")(_model(axes))
    conditions = tuple(
        phx.conditions.Periodic(
            "u",
            phx.domain.PeriodicIdentification(box, "x", component=axis).pairing(),
        )
        for axis in range(axes)
    )
    resources = phx.enforcement.PeriodicResourcePolicy(maximum_endpoint_evaluations=limit)
    try:
        phx.enforcement.prepare_periodic_projection(
            {"u": u}, conditions, route="analytic", resources=resources
        )
    except ValueError as error:
        return {
            "axes": axes,
            "maximum_endpoint_evaluations": limit,
            "status": "refused",
            "message": str(error),
        }
    raise RuntimeError("The endpoint-evaluation limit was not enforced.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--orders", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--queries", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--reference-queries", type=int, default=1024)
    parser.add_argument("--reference-axes", type=int, default=2)
    parser.add_argument("--reference-order", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/periodic_enforcement.json")
    )
    args = parser.parse_args()
    if (
        args.repeats < 1
        or any(value < 1 for value in (*args.axes, *args.queries))
        or any(value < 0 for value in args.orders)
    ):
        raise ValueError(
            "Axes, queries, and repeats must be positive; orders nonnegative."
        )
    jax.config.update("jax_enable_x64", True)
    # Order/axis sweep at the reference query count, then a query sweep at the
    # reference axis count and order; the shared reference case runs once.
    grid = [
        (axes, order, args.reference_queries)
        for axes in args.axes
        for order in args.orders
    ]
    grid.extend(
        (args.reference_axes, args.reference_order, queries)
        for queries in args.queries
        if (args.reference_axes, args.reference_order, queries) not in grid
    )
    cases = [_case(*row, args.repeats) for row in grid]
    refusals = [_refusal(3, 9)]
    identity = capture_benchmark_identity(
        Path(__file__).resolve().parent.parent, Path(__file__).resolve(), cases[0].keys()
    )
    report = {
        "kind": "periodic-enforcement-capacity-campaign",
        "identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "refusals": refusals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

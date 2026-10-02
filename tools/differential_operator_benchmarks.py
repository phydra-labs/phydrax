# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.operators.differential import (
    evaluate_fused_coordinate_derivatives,
    evaluate_taylor_contractions,
    plan_taylor_contractions,
    TaylorContractionPlan,
    TaylorContractionPolicy,
    TaylorContractionRequest,
    TaylorContractionResources,
    TaylorContractionStrategy,
)
from phydrax.operators.differential._jet import jet_dn


def measure_kernel(
    function: Callable[..., Array],
    arguments: tuple[Array, ...],
    *,
    repeats: int,
    retained_bytes: int | None = None,
    working_bytes: int | None = None,
) -> tuple[Array, dict[str, Any]]:
    """One stable callable per capacity; compiler bytes are not logical estimates."""
    jitted, binding_seconds = measure_host(lambda: jax.jit(function))
    compiled, phases = measure_lower_and_compile(
        lambda: jitted.lower(*arguments), lambda lowered: lowered.compile()
    )
    value, first_seconds = measure_synchronized(lambda: compiled(*arguments))
    _, warmed = measure_repeated(lambda: compiled(*arguments), warmup=1, repeats=repeats)
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The active JAX backend does not report every compiler estimate.",
    )
    return value, {
        "binding_seconds": binding_seconds,
        "lowering_seconds": phases.lowering_seconds,
        "compilation_seconds": phases.compilation_seconds,
        "first_synchronized_execution_seconds": first_seconds,
        "warmed": warmed.to_seconds_dict(),
        "compiler": asdict(evidence),
        "logical_retained_bytes": logical_array_bytes((arguments, value))
        if retained_bytes is None
        else retained_bytes + logical_array_bytes(value),
        "logical_retained_scope": "Dynamic input, captured numerical leaves, and result arrays; excludes static host plan metadata.",
        "logical_working_bytes_estimate": working_bytes,
        "logical_working_scope": "Estimated live numerical curve/model series only; not a certified total bound, and excludes compiler temporaries and output accumulators.",
    }


def _jvp_contraction(
    function: Callable[[Array], Array],
    directions: tuple[Array, ...],
    multiplicities: tuple[int, ...],
) -> Callable[[Array], Array]:
    current = function
    for direction, count in zip(directions, multiplicities, strict=True):
        for _ in range(count):
            previous = current

            def derivative(
                point: Array,
                previous: Callable[[Array], Array] = previous,
                direction: Array = direction,
            ) -> Array:
                return jax.jvp(previous, (point,), (direction,))[1]

            current = derivative
    return current


def _legacy_contraction(
    function: Callable[[Array], Array],
    directions: tuple[Array, ...],
    multiplicities: tuple[int, ...],
) -> Callable[[Array], Array]:
    current = function
    for direction, count in zip(directions, multiplicities, strict=True):
        previous = current

        def derivative(
            point: Array,
            previous: Callable[[Array], Array] = previous,
            direction: Array = direction,
            count: int = count,
        ) -> Array:
            return jet_dn(previous, point, direction, n=count)

        current = derivative
    return current


def _output(value: Array, kind: str) -> Array:
    if kind == "scalar":
        return value
    if kind == "vector":
        return jnp.stack((value, 2 * value))
    if kind == "complex":
        complex_dtype = jnp.complex128 if value.dtype == jnp.float64 else jnp.complex64
        return value.astype(complex_dtype) * jnp.asarray(1 + 0.5j, dtype=complex_dtype)
    raise ValueError(f"Unknown output kind {kind!r}.")


def _plan(
    counts: tuple[int, ...],
    strategy: TaylorContractionStrategy,
    workset: int,
) -> TaylorContractionPlan:
    request = TaylorContractionRequest(
        tuple(f"direction-{index}" for index in range(len(counts))), counts
    )
    return plan_taylor_contractions(
        (request,),
        policy=TaylorContractionPolicy(
            strategy=strategy,
            resources=TaylorContractionResources(workset_size=workset),
        ),
    )


def _contraction_case(
    dimension: int,
    counts: tuple[int, ...],
    kind: str,
    strategy: TaylorContractionStrategy,
    dtype: DTypeLike,
    probes: int,
    workset: int,
    repeats: int,
    reference_order_limit: int,
    gradient_order_limit: int,
) -> dict[str, Any]:
    plan, planning_seconds = measure_host(lambda: _plan(counts, strategy, workset))
    point = jnp.linspace(0.2, 0.8, dimension, dtype=dtype)
    coefficients = jnp.linspace(0.7, 1.3, dimension, dtype=dtype)
    directions = jnp.stack(
        tuple(
            jnp.linspace(0.1 + index * 0.02, 0.3 + index * 0.02, dimension, dtype=dtype)
            for index in range(len(counts))
        )
    )
    points = jnp.stack(tuple(point + index * 0.001 for index in range(probes)))
    degree = sum(counts) + 2

    def field(weights: Array, position: Array) -> Array:
        return _output(jnp.sum(weights * position**degree), kind)

    def operation(weights: Array, positions: Array, tangents: Array) -> Array:
        def at_point(position: Array) -> Array:
            def bound(value: Array) -> Array:
                return field(weights, value)

            return evaluate_taylor_contractions(
                bound,
                (position,),
                {
                    identifier: (tangents[index],)
                    for index, identifier in enumerate(plan.direction_ids)
                },
                plan,
            ).values[0]

        return jax.vmap(at_point)(positions)

    retained = logical_array_bytes((coefficients, points, directions))
    # Curve input/output coefficients only; excludes model intermediates/compiler temps.
    working = (
        probes
        * min(workset, len(plan.schedules))
        * (plan.required_regularity_order + 1)
        * (dimension + (2 if kind == "vector" else 1))
        * np.dtype(dtype).itemsize
        * (2 if kind == "complex" else 1)
    )
    actual, timings = measure_kernel(
        operation,
        (coefficients, points, directions),
        repeats=repeats,
        retained_bytes=retained,
        working_bytes=working,
    )
    host_points = np.asarray(jax.device_get(points))
    product = np.prod(
        np.stack(
            tuple(
                np.asarray(jax.device_get(directions[index])) ** count
                for index, count in enumerate(counts)
            )
        ),
        axis=0,
    )
    reference = np.sum(
        np.asarray(jax.device_get(coefficients))
        * host_points ** (degree - sum(counts))
        * product,
        axis=-1,
    ) * (math.factorial(degree) // math.factorial(degree - sum(counts)))
    if kind == "vector":
        reference = np.stack((reference, 2 * reference), axis=-1)
    elif kind == "complex":
        reference = reference * (1 + 0.5j)

    def bound(position: Array) -> Array:
        return field(coefficients, position)

    tangent_tuple = tuple(directions[index] for index in range(len(counts)))
    comparisons: dict[str, Any] = {}
    if sum(counts) <= reference_order_limit:
        for name, comparison in (
            ("nested_jvp", _jvp_contraction(bound, tangent_tuple, counts)),
            ("legacy_jet", _legacy_contraction(bound, tangent_tuple, counts)),
        ):
            value, timing = measure_kernel(
                jax.vmap(comparison), (points,), repeats=repeats, retained_bytes=retained
            )
            comparisons[name] = timing | {
                "maximum_absolute_error": float(
                    np.max(np.abs(np.asarray(jax.device_get(value)) - reference))
                )
            }
    else:
        comparisons["not_measured_reason"] = (
            "Derivative order exceeds the explicit bounded reference campaign."
        )

    def objective(weights: Array, tangents: Array) -> Array:
        values = operation(weights, points, tangents)
        return jnp.mean(jnp.real(values * jnp.conj(values)))

    if plan.required_regularity_order > gradient_order_limit:
        gradient_evidence: dict[str, Any] = {
            "admitted": False,
            "reason": "Executed scalar Jet order exceeds the gradient compilation budget.",
            "executed_order": plan.required_regularity_order,
            "maximum_executed_order": gradient_order_limit,
        }
        direction_evidence = dict(gradient_evidence)
    else:
        gradient, gradient_timing = measure_kernel(
            jax.grad(objective, argnums=0),
            (coefficients, directions),
            repeats=repeats,
            retained_bytes=retained,
        )
        direction_gradient, direction_timing = measure_kernel(
            jax.grad(objective, argnums=1),
            (coefficients, directions),
            repeats=repeats,
            retained_bytes=retained,
        )
        gradient_evidence = gradient_timing | {
            "admitted": True,
            "norm": float(jnp.linalg.norm(gradient)),
        }
        direction_evidence = direction_timing | {
            "admitted": True,
            "norm": float(jnp.linalg.norm(direction_gradient)),
        }
    return {
        "case": f"{kind}-{'pure' if len(counts) == 1 else 'mixed'}",
        "dimension": dimension,
        "multiplicities": counts,
        "strategy": strategy,
        "probe_count": probes,
        "workset": workset,
        "dtype": str(np.dtype(dtype)),
        "plan_id": plan.plan_id,
        "schedule_count": len(plan.schedules),
        "executed_order": plan.required_regularity_order,
        "host_planning_certification_seconds": planning_seconds,
        "maximum_absolute_error": float(
            np.max(np.abs(np.asarray(jax.device_get(actual)) - reference))
        ),
        "execution": timings,
        "comparisons": comparisons,
        "parameter_gradient": gradient_evidence,
        "direction_gradient": direction_evidence,
    }


def _model_case(
    dimension: int,
    width: int,
    depth: int,
    order: int,
    dtype: DTypeLike,
    probes: int,
    workset: int,
    repeats: int,
    reference_order_limit: int,
) -> dict[str, Any]:
    plan, planning_seconds = measure_host(lambda: _plan((order,), "linear", workset))
    point = jnp.linspace(-0.2, 0.2, dimension, dtype=dtype)
    points = jnp.broadcast_to(point, (probes, dimension))
    direction = jnp.full((dimension,), 1 / dimension, dtype=dtype)
    input_weights = jnp.full((width, dimension), 0.2 / dimension, dtype=dtype)
    hidden_weights = jnp.full((depth, width, width), 0.2 / width, dtype=dtype)

    def field(position: Array, inputs: Array, hidden: Array) -> Array:
        value = jnp.tanh(inputs @ position + jnp.asarray(0.1, dtype=dtype))
        for index in range(depth):
            value = jnp.tanh(hidden[index] @ value + jnp.asarray(0.1, dtype=dtype))
        return jnp.mean(value)

    def operation(inputs: Array, hidden: Array, positions: Array) -> Array:
        def at_point(position: Array) -> Array:
            def bound(value: Array) -> Array:
                return field(value, inputs, hidden)

            return evaluate_taylor_contractions(
                bound, (position,), {"direction-0": (direction,)}, plan
            ).values[0]

        return jax.vmap(at_point)(positions)

    value, timing = measure_kernel(
        operation,
        (input_weights, hidden_weights, points),
        repeats=repeats,
        retained_bytes=logical_array_bytes(
            (input_weights, hidden_weights, points, direction)
        ),
        working_bytes=probes
        * (order + 1)
        * (dimension + width * (depth + 1))
        * np.dtype(dtype).itemsize,
    )

    def bound(position: Array) -> Array:
        return field(position, input_weights, hidden_weights)

    jvp_error = None
    if order <= reference_order_limit:
        reference = jax.vmap(_jvp_contraction(bound, (direction,), (order,)))(points)
        jvp_error = float(jnp.max(jnp.abs(value - reference)))

    def loss(inputs: Array, hidden: Array, positions: Array) -> Array:
        return jnp.mean(operation(inputs, hidden, positions) ** 2)

    _, gradient_timing = measure_kernel(
        jax.grad(loss, argnums=1),
        (input_weights, hidden_weights, points),
        repeats=repeats,
        retained_bytes=logical_array_bytes(
            (input_weights, hidden_weights, points, direction)
        ),
    )
    return {
        "case": "model-capacity",
        "dimension": dimension,
        "width": width,
        "depth": depth,
        "order": order,
        "probe_count": probes,
        "host_planning_certification_seconds": planning_seconds,
        "maximum_absolute_jvp_error": jvp_error,
        "execution": timing,
        "parameter_gradient": gradient_timing,
    }


def _dense_reference(dimension: int, dtype: DTypeLike, repeats: int) -> dict[str, Any]:
    point = jnp.linspace(-0.7, 0.9, dimension, dtype=dtype)

    def function(value: Array) -> Array:
        return jnp.sum(jnp.sin(value) + jnp.asarray(0.05, dtype=dtype) * value**3)

    def action(value: Array) -> Array:
        evaluated = evaluate_fused_coordinate_derivatives(
            function, value, second_axes=tuple(range(dimension))
        )
        return sum(evaluated.diagonal_second_derivatives, jnp.asarray(0, dtype=dtype))

    def dense(value: Array) -> Array:
        return jnp.trace(jax.hessian(function)(value))

    action_value, action_timing = measure_kernel(action, (point,), repeats=repeats)
    dense_value, dense_timing = measure_kernel(dense, (point,), repeats=repeats)
    return {
        "case": "bounded-coordinate-laplacian",
        "dimension": dimension,
        "absolute_error": float(jnp.abs(action_value - dense_value)),
        "action": action_timing,
        "dense": dense_timing,
    }


def _integers(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item < 1 for item in values):
        raise ValueError("Capacities must be comma-separated positive integers.")
    return values


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Capacity-controlled exact Taylor contractions; no global speedup claim."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--dimensions", type=_integers, default=(16,))
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--multiplicities", type=_integers, default=(2, 2))
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--workset", type=int, default=4)
    parser.add_argument("--widths", type=_integers, default=(8,))
    parser.add_argument("--depths", type=_integers, default=(1,))
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dense-limit", type=int, default=32)
    parser.add_argument("--reference-order-limit", type=int, default=4)
    parser.add_argument(
        "--gradient-order-limit",
        type=int,
        default=16,
        help="Refuse reverse-gradient qualification above this executed Jet order.",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    dimensions = (
        (arguments.dimension,)
        if arguments.dimension is not None
        else arguments.dimensions
    )
    order, counts = arguments.order, arguments.multiplicities
    probes, workset, repeats = arguments.probes, arguments.workset, arguments.repeats
    widths, depths = arguments.widths, arguments.depths
    if arguments.smoke:
        dimensions, order, counts, probes, workset, repeats, widths, depths = (
            (3,),
            2,
            (1, 1),
            2,
            2,
            2,
            (2,),
            (1,),
        )
    if any(item < 1 for item in (*dimensions, order, probes, workset, repeats)):
        raise ValueError("All capacities must be positive.")
    if not 0 <= arguments.dense_limit <= 64:
        raise ValueError(
            "dense-limit must be between zero and 64; dense references are bounded."
        )
    if not 0 <= arguments.reference_order_limit <= 8:
        raise ValueError("reference-order-limit must be between zero and eight.")
    jax.config.update("jax_enable_x64", arguments.dtype == "float64")
    dtype = np.dtype(arguments.dtype)
    records: list[dict[str, Any]] = []
    strategies: tuple[TaylorContractionStrategy, ...] = ("linear", "single", "prime")
    for dimension in dimensions:
        for multiplicities in ((order,), counts):
            for kind in ("scalar", "vector", "complex"):
                for strategy in strategies:
                    records.append(
                        _contraction_case(
                            dimension,
                            multiplicities,
                            kind,
                            strategy,
                            dtype,
                            probes,
                            workset,
                            repeats,
                            arguments.reference_order_limit,
                            arguments.gradient_order_limit,
                        )
                    )
                    # Each record owns cold compilation plus its own warm samples.
                    # Do not retain every executable across a capacity campaign.
                    jax.clear_caches()
        for width in widths:
            for depth in depths:
                records.append(
                    _model_case(
                        dimension,
                        width,
                        depth,
                        order,
                        dtype,
                        probes,
                        workset,
                        repeats,
                        arguments.reference_order_limit,
                    )
                )
                jax.clear_caches()
        if dimension <= arguments.dense_limit:
            records.append(_dense_reference(dimension, dtype, repeats))
            jax.clear_caches()
    driver = Path(__file__).resolve()
    payload = {
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            driver.parent.parent, driver, ("records", "environment")
        ).to_dict(),
        "records": records,
        "interpretation": "Capacity-local measurements; logical coefficient buffers exclude compiler temporary memory. Analytic monomials are independent references.",
    }
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    if arguments.output is not None:
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()

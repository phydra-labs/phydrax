#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from high_order_conservation_qualification import (
    _smooth_periodic_state,
    _tensor_problem,
    _triangle_problem,
)


def _timed_samples(compiled, arguments, minimum_seconds, minimum_repeats):
    samples = []
    repeat_count = int(minimum_repeats)
    for _trial in range(5):
        while True:
            start = perf_counter()
            result = None
            for _ in range(repeat_count):
                result = compiled(*arguments)
            jax.block_until_ready(result)
            elapsed = perf_counter() - start
            if elapsed >= minimum_seconds:
                break
            repeat_count *= 2
        samples.append(elapsed / repeat_count)
    return result, repeat_count, np.asarray(samples)


def _measure_callable(
    function,
    arguments,
    dof_count,
    resident_bytes,
    *,
    minimum_seconds,
    minimum_repeats,
):
    compiled = jax.jit(function)
    start = perf_counter()
    result = compiled(*arguments)
    jax.block_until_ready(result)
    compile_seconds = perf_counter() - start
    result, repeat_count, samples = _timed_samples(
        compiled, arguments, minimum_seconds, minimum_repeats
    )
    median = float(np.median(samples))
    variation = float(np.std(samples) / np.mean(samples))
    return {
        "dofs": int(dof_count),
        "compile_seconds": compile_seconds,
        "seconds_per_call_median": median,
        "seconds_per_call_samples": samples.tolist(),
        "repeats_per_trial": int(repeat_count),
        "dof_updates_per_second": float(dof_count / median),
        "coefficient_of_variation": variation,
        "resident_bytes_per_dof": float(resident_bytes / dof_count),
        "finite": bool(jnp.all(jnp.isfinite(result))),
    }


def _measure_dynamics(
    dynamics,
    state,
    *,
    minimum_seconds,
    minimum_repeats,
    ad_dynamics=None,
    ad_state=None,
):
    derivative_dynamics = dynamics if ad_dynamics is None else ad_dynamics
    derivative_state = state if ad_state is None else ad_state
    direction = jnp.linspace(
        -0.25,
        0.25,
        derivative_state.size,
        dtype=derivative_state.dtype,
    ).reshape(derivative_state.shape)
    cotangent = jnp.linspace(
        0.2,
        -0.15,
        derivative_state.size,
        dtype=derivative_state.dtype,
    ).reshape(derivative_state.shape)
    primal = _measure_callable(
        lambda value: dynamics(0.0, value),
        (state,),
        state.size,
        state.nbytes * 2,
        minimum_seconds=minimum_seconds,
        minimum_repeats=minimum_repeats,
    )
    ad_primal = _measure_callable(
        lambda value: derivative_dynamics(0.0, value),
        (derivative_state,),
        derivative_state.size,
        derivative_state.nbytes * 2,
        minimum_seconds=minimum_seconds,
        minimum_repeats=minimum_repeats,
    )
    jvp = _measure_callable(
        lambda value, tangent: jax.jvp(
            lambda candidate: derivative_dynamics(0.0, candidate),
            (value,),
            (tangent,),
        )[1],
        (derivative_state, direction),
        derivative_state.size,
        derivative_state.nbytes * 3,
        minimum_seconds=minimum_seconds,
        minimum_repeats=minimum_repeats,
    )
    vjp = _measure_callable(
        lambda value, dual: jax.vjp(
            lambda candidate: derivative_dynamics(0.0, candidate), value
        )[1](dual)[0],
        (derivative_state, cotangent),
        derivative_state.size,
        derivative_state.nbytes * 3,
        minimum_seconds=minimum_seconds,
        minimum_repeats=minimum_repeats,
    )
    jvp["cost_ratio_to_primal"] = (
        jvp["seconds_per_call_median"] / ad_primal["seconds_per_call_median"]
    )
    vjp["cost_ratio_to_primal"] = (
        vjp["seconds_per_call_median"] / ad_primal["seconds_per_call_median"]
    )
    return {
        "primal": primal,
        "ad_primal": ad_primal,
        "jvp": jvp,
        "vjp": vjp,
    }


def _case_passed(case, limits):
    primal = case["primal"]
    measured = ("primal", "ad_primal", "jvp", "vjp")
    return bool(
        all(case[name]["finite"] for name in measured)
        and all(
            case[name]["compile_seconds"] <= limits["maximum_compile_seconds"]
            for name in measured
        )
        and primal["seconds_per_call_median"] <= limits["maximum_rhs_seconds"]
        and primal["dof_updates_per_second"] >= limits["minimum_dof_updates_per_second"]
        and all(
            case[name]["coefficient_of_variation"] <= limits["maximum_variation"]
            for name in measured
        )
        and all(
            case[name]["resident_bytes_per_dof"]
            <= limits["maximum_resident_bytes_per_dof"]
            for name in measured
        )
        and case["jvp"]["cost_ratio_to_primal"] <= limits["maximum_jvp_ratio"]
        and case["vjp"]["cost_ratio_to_primal"] <= limits["maximum_vjp_ratio"]
    )


def run(
    *,
    profile="production",
    minimum_seconds=1.0,
    minimum_repeats=5,
    limits=None,
):
    if profile == "production":
        tensor_shape = (32, 32, 4)
        triangle_shape = (48, 48, 2)
    elif profile == "smoke":
        tensor_shape = (8, 8, 2)
        triangle_shape = (8, 8, 2)
    else:
        raise ValueError("Unknown high-order benchmark profile.")
    nx, ny, tensor_degree = tensor_shape
    tensor, tensor_system, tensor_discretization = _tensor_problem(
        physical_boundaries=False,
        nx=nx,
        ny=ny,
        degree=tensor_degree,
    )
    tensor_state = _smooth_periodic_state(tensor_system, tensor_discretization)
    viscous, viscous_system, viscous_discretization = _tensor_problem(
        viscous=True,
        physical_boundaries=False,
        nx=nx,
        ny=ny,
        degree=tensor_degree,
    )
    viscous_state = _smooth_periodic_state(viscous_system, viscous_discretization)
    tri_nx, tri_ny, triangle_degree = triangle_shape
    triangle, triangle_system, triangle_discretization = _triangle_problem(
        tri_nx, tri_ny, triangle_degree
    )
    triangle_state = _smooth_periodic_state(triangle_system, triangle_discretization)
    if profile == "production":
        tensor_ad, tensor_ad_system, tensor_ad_discretization = _tensor_problem(
            physical_boundaries=False, nx=8, ny=8, degree=2
        )
        tensor_ad_state = _smooth_periodic_state(
            tensor_ad_system, tensor_ad_discretization
        )
        viscous_ad, viscous_ad_system, viscous_ad_discretization = _tensor_problem(
            viscous=True,
            physical_boundaries=False,
            nx=8,
            ny=8,
            degree=2,
        )
        viscous_ad_state = _smooth_periodic_state(
            viscous_ad_system, viscous_ad_discretization
        )
        triangle_ad, triangle_ad_system, triangle_ad_discretization = _triangle_problem(
            8, 8, 2
        )
        triangle_ad_state = _smooth_periodic_state(
            triangle_ad_system, triangle_ad_discretization
        )
    else:
        tensor_ad, tensor_ad_state = tensor, tensor_state
        viscous_ad, viscous_ad_state = viscous, viscous_state
        triangle_ad, triangle_ad_state = triangle, triangle_state
    selected_limits = {
        "maximum_compile_seconds": 600.0,
        "maximum_rhs_seconds": 2.0,
        "minimum_dof_updates_per_second": 1.0e3,
        "maximum_variation": 0.10,
        "maximum_resident_bytes_per_dof": 512.0,
        "maximum_jvp_ratio": 3.0,
        "maximum_vjp_ratio": 5.0,
        **({} if limits is None else limits),
    }
    cases = {}
    cases["tensor_dgsem"] = _measure_dynamics(
        tensor,
        tensor_state,
        minimum_seconds=minimum_seconds,
        minimum_repeats=minimum_repeats,
        ad_dynamics=tensor_ad,
        ad_state=tensor_ad_state,
    )
    jax.clear_caches()
    gc.collect()
    cases["tensor_viscous_dg"] = _measure_dynamics(
        viscous,
        viscous_state,
        minimum_seconds=minimum_seconds,
        minimum_repeats=minimum_repeats,
        ad_dynamics=viscous_ad,
        ad_state=viscous_ad_state,
    )
    jax.clear_caches()
    gc.collect()
    cases["triangle_nodal_dg"] = _measure_dynamics(
        triangle,
        triangle_state,
        minimum_seconds=minimum_seconds,
        minimum_repeats=minimum_repeats,
        ad_dynamics=triangle_ad,
        ad_state=triangle_ad_state,
    )
    jax.clear_caches()
    gc.collect()
    result = {
        "profile": profile,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "dtype": str(tensor_state.dtype),
        "limits": selected_limits,
        "cases": cases,
    }
    result["passed"] = all(_case_passed(case, selected_limits) for case in cases.values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("smoke", "production"), default="production"
    )
    parser.add_argument("--minimum-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/high_order_conservation.json"),
    )
    args = parser.parse_args()
    if args.minimum_seconds <= 0.0 or args.minimum_repeats <= 0:
        raise ValueError("Benchmark sample controls must be positive.")
    result = run(
        profile=args.profile,
        minimum_seconds=args.minimum_seconds,
        minimum_repeats=args.minimum_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("High-order conservation benchmark failed its SLOs.")


if __name__ == "__main__":
    main()

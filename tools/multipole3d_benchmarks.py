#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.operators.integral.multipole._laplace3d import _LaplaceTranslationData


def _preparation(plan: Any) -> tuple[Any, dict[str, float]]:
    if isinstance(plan, phx.operators.LaplaceMultipolePlan3D):
        data, table_seconds = measure_synchronized(lambda: _LaplaceTranslationData(plan))
        prepared, topology_seconds = measure_synchronized(
            lambda: plan.prepare(_translation_data=data)
        )
        return prepared, {
            "translation_tables_seconds": table_seconds,
            "topology_and_binding_seconds": topology_seconds,
            "combined_seconds": table_seconds + topology_seconds,
        }
    prepared, preparation_seconds = measure_synchronized(plan.prepare)
    return prepared, {"topology_and_translation_tables_seconds": preparation_seconds}


def _problem(source_count: int, target_count: int) -> Any:
    rng = np.random.default_rng(390174)
    sources = jnp.asarray(
        rng.uniform(-0.9, 0.9, size=(source_count, 3)), dtype=jnp.float64
    )
    targets = jnp.asarray(
        rng.uniform(-0.9, 0.9, size=(target_count, 3)), dtype=jnp.float64
    )
    strengths = jnp.asarray(rng.normal(size=(source_count,)), dtype=jnp.float64)
    return sources, targets, strengths


def _reference(
    kernel: str, sources: Any, strengths: Any, targets: Any, parameter: float
) -> Any:
    radii = jnp.linalg.norm(targets[:, None, :] - sources[None, :, :], axis=-1)
    if kernel == "laplace":
        numerator = jnp.ones_like(radii)
    elif kernel == "helmholtz":
        numerator = jnp.exp(1j * parameter * radii)
    else:
        numerator = jnp.exp(-parameter * radii)
    return jnp.sum(numerator * strengths[None, :] / (4.0 * jnp.pi * radii), axis=1)


def _phase(
    function: Any, arguments: tuple[Any, ...], repeats: int
) -> tuple[Any, dict[str, Any]]:
    """Keep lowering, compile, first execution, and warmed execution distinct."""
    staged: Any = eqx.filter_jit(function)
    compiled, compilation = measure_lower_and_compile(
        lambda: staged.lower(*arguments), lambda lowered: lowered.compile()
    )
    first, first_seconds = measure_synchronized(lambda: compiled(*arguments))
    result, distribution = measure_repeated(
        lambda: compiled(*arguments), warmup=1, repeats=repeats
    )
    return result, {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_execution_seconds": first_seconds,
        "warmed_execution": distribution.to_seconds_dict(),
        "checksum": float(
            sum(
                jnp.sum(jnp.abs(leaf))
                for leaf in jax.tree.leaves(first)
                if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.inexact)
            )
        ),
    }


def _translation_phases(
    prepared: Any, sources: Any, strengths: Any, repeats: int
) -> dict[str, Any]:
    source_center = jnp.asarray([-0.2, 0.1, -0.1], dtype=jnp.float64)
    target_center = jnp.asarray([0.4, -0.3, 0.5], dtype=jnp.float64)
    multipole = prepared.p2m(sources, strengths, source_center)
    _, m2m_phase = _phase(
        prepared.m2m, (multipole, source_center, source_center + 0.1), repeats
    )
    local, m2l_phase = _phase(
        prepared.m2l, (multipole, source_center, target_center), repeats
    )
    _, l2l_phase = _phase(
        prepared.l2l, (local, target_center, target_center + 0.02), repeats
    )
    direction = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float64)

    def translation_jvp(center: Any) -> Any:
        return jax.jvp(
            lambda shift: prepared.m2l(multipole, source_center, shift),
            (center,),
            (direction,),
        )[1]

    _, derivative = _phase(translation_jvp, (target_center,), repeats)
    return {
        "m2m": m2m_phase,
        "m2l": m2l_phase,
        "l2l": l2l_phase,
        "m2l_geometry_jvp": derivative,
    }


def _case(
    kernel: str,
    sources: Any,
    targets: Any,
    strengths: Any,
    order: int,
    depth: int,
    repeats: int,
    parameter: float,
    translation_route: str,
) -> Any:
    if kernel == "laplace":
        plan_type, keyword = phx.operators.LaplaceMultipolePlan3D, {}
    elif kernel == "helmholtz":
        plan_type, keyword = (
            phx.operators.HelmholtzMultipolePlan3D,
            {"wavenumber": parameter},
        )
    else:
        plan_type, keyword = (
            phx.operators.ModifiedHelmholtzMultipolePlan3D,
            {"decay": parameter},
        )
    plan, policy_seconds = measure_synchronized(
        lambda: plan_type(
            sources,
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            reference_targets=targets,
            depth=depth,
            expansion_order=order,
            translation_route=translation_route,
            **keyword,
        )
    )
    prepared, preparation = _preparation(plan)
    result, evaluation = _phase(
        lambda source, weight, target: prepared.evaluate(source, weight, target),
        (sources, strengths, targets),
        repeats,
    )

    def strength_vjp(source: Any, weight: Any, target: Any) -> Any:
        return jax.grad(
            lambda strengths_: jnp.real(
                jnp.sum(prepared.evaluate(source, strengths_, target).values)
            )
        )(weight)

    _, derivative = _phase(strength_vjp, (sources, strengths, targets), repeats)
    expected = _reference(kernel, sources, strengths, targets, parameter)
    resources = prepared.resources
    return {
        "translation_route": translation_route,
        "plan_id": plan.plan_id,
        "prepared_id": prepared.prepared_id,
        "preparation": {
            "policy_and_reference_transfer_seconds": policy_seconds,
            **preparation,
        },
        "translations": _translation_phases(prepared, sources, strengths, repeats),
        "complete_evaluation": evaluation,
        "complete_strength_vjp": derivative,
        "maximum_absolute_error": float(jnp.max(jnp.abs(result.values - expected))),
        "successful": bool(result.successful),
        "logical_modes": resources.logical_mode_count,
        "padded_modes": resources.padded_mode_count,
        "required_coefficient_bytes": resources.required_coefficient_bytes,
        "maximum_coefficient_bytes": resources.maximum_coefficient_bytes,
        "translation_table_bytes": resources.translation_table_bytes,
        "translation_workspace_bytes": resources.translation_workspace_bytes,
        "quadrature_nodes": resources.quadrature_node_count,
        "far_interaction_capacity": resources.far_interaction_capacity,
        "near_interaction_capacity": resources.near_interaction_capacity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark complete 3-D multipole passes."
    )
    parser.add_argument("--source-count", type=int, default=16)
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parameter", type=float, default=0.7)
    parser.add_argument(
        "--kernel",
        choices=("all", "laplace", "helmholtz", "modified-helmholtz"),
        default="all",
    )
    parser.add_argument(
        "--translation-route", choices=("dense", "rotation"), default="dense"
    )
    arguments = parser.parse_args()
    if arguments.translation_route == "rotation" and arguments.kernel != "laplace":
        parser.error(
            "--translation-route rotation requires --kernel laplace; screened kernels refuse this route"
        )
    sources, targets, strengths = _problem(arguments.source_count, arguments.target_count)
    kernels = (
        ("laplace", "helmholtz", "modified-helmholtz")
        if arguments.kernel == "all"
        else (arguments.kernel,)
    )
    output = {
        "source_count": arguments.source_count,
        "target_count": arguments.target_count,
        "expansion_order": arguments.order,
        "depth": arguments.depth,
        "parameter": arguments.parameter,
        "translation_route": arguments.translation_route,
        "environment": capture_environment().to_dict(),
        "kernels": {
            kernel: _case(
                kernel,
                sources,
                targets,
                strengths,
                arguments.order,
                arguments.depth,
                arguments.repeats,
                arguments.parameter,
                arguments.translation_route,
            )
            for kernel in kernels
        },
    }
    root = Path(__file__).resolve().parents[1]
    output["identity"] = capture_benchmark_identity(
        root, Path(__file__), output.keys()
    ).to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Bounded kernel correction workspace; run with --smoke for small cases."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax import config as pxconfig
from phydrax.conditions import ArrayCodomain, FieldSpec, ProductFieldSpec
from phydrax.kernels import (
    KernelFunctional,
    KernelFunctionalTerm,
    ProductFieldKernelMetric,
    SquaredExponentialKernel,
)
from phydrax.linalg import (
    ArraySpace,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    solve,
    TolerancePolicy,
)

from ._io import write_json_atomic
from ._memory import fitted_scaling_exponent, host_allocation_peak, process_peak_rss_bytes
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


class _Executable(Protocol):
    compiled: jax.stages.Compiled

    def __call__(
        self, metric: ProductFieldKernelMetric, functional: KernelFunctional
    ) -> tuple[Array, Array, Array]: ...


class _Lowered(Protocol):
    def compile(self) -> _Executable: ...


@runtime_checkable
class _Entry(Protocol):
    def __call__(
        self, metric: ProductFieldKernelMetric, functional: KernelFunctional
    ) -> tuple[Array, Array, Array]: ...

    def lower(
        self, metric: ProductFieldKernelMetric, functional: KernelFunctional
    ) -> _Lowered: ...


def _require_entry(
    entry: Callable[
        [ProductFieldKernelMetric, KernelFunctional], tuple[Array, Array, Array]
    ],
) -> _Entry:
    if not isinstance(entry, _Entry):
        raise TypeError("Kernel correction benchmark requires a lowering entry.")
    return entry


def _probes(
    metric: ProductFieldKernelMetric,
    functional: KernelFunctional,
    block: int,
    candidate: bool,
) -> tuple[Array, Array, Array]:
    count = functional.row_count

    def action(vector: Array) -> Array:
        if candidate:
            return metric.functional_gram_action(
                functional, vector, row_block_size=block, column_block_size=block
            )
        return metric.functional_gram(functional).matrix @ vector

    space = ArraySpace((count,), dtype=jnp.float64)
    operator = FunctionLinearOperator(
        action,
        source=space,
        target=space,
        properties=OperatorProperties(self_adjoint=True, positive_semidefinite=True),
        operator_id="memory-kernel-gram",
    )
    prepared = prepare(
        LinearSystem(operator),
        LinearSolvePolicy(
            GMRES(restart=8),
            tolerance=TolerancePolicy(relative=1e-9, absolute=1e-11, max_steps=64),
        ),
    )
    if not candidate:
        identity = jnp.eye(count, dtype=jnp.float64)
        result = solve(prepared, identity)
        inverse = jnp.asarray(result.value)
        defect = jnp.max(
            jnp.abs(jax.vmap(operator.mv, in_axes=-1, out_axes=-1)(inverse) - identity)
        )
        return (
            defect,
            metric.functional_gram(functional).matrix.diagonal(),
            jnp.all(result.successful),
        )

    def probe(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        defect, successful = carry
        columns = index * block + jnp.arange(block, dtype=jnp.int32)
        identity = (
            jnp.arange(count, dtype=jnp.int32)[:, None] == columns[None, :]
        ).astype(jnp.float64)
        result = solve(prepared, identity)
        applied = action(jnp.asarray(result.value))
        return jnp.maximum(
            defect, jnp.max(jnp.abs(applied - identity))
        ), successful & jnp.all(result.successful)

    defect, successful = jax.lax.fori_loop(
        0,
        (count + block - 1) // block,
        probe,
        (jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(True)),
    )
    from phydrax.kernels import kernel_functional_diagonal

    return (
        defect,
        kernel_functional_diagonal(
            metric, functional, row_block_size=block, column_block_size=block
        ),
        successful,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
    sizes = [8, 16] if args.smoke else args.sizes
    cases: list[dict[str, Any]] = []
    for count in sizes:
        spec = ProductFieldSpec(
            (FieldSpec("u", ArrayCodomain.from_shape((), dtype="float64")),)
        )
        metric = ProductFieldKernelMetric.independent(
            spec, {"u": SquaredExponentialKernel(length_scale=0.2)}
        )
        points = jnp.arange(count, dtype=jnp.float64)[:, None]
        coefficients = jnp.eye(count, dtype=jnp.float64)[:, :, None, None]
        functional = KernelFunctional(
            (KernelFunctionalTerm("u", points, ((0,),), coefficients),),
            functional_id="memory-kernel-functional",
        )
        entry = _require_entry(
            eqx.filter_jit(
                partial(_probes, block=args.block, candidate=not args.baseline)
            )
        )
        (executable, timing), host_peak = host_allocation_peak(
            lambda: measure_lower_and_compile(
                lambda: entry.lower(metric, functional), lambda lowered: lowered.compile()
            )
        )
        result, warm = measure_repeated(
            lambda: executable(metric, functional), warmup=1, repeats=2
        )
        evidence = compiler_evidence(
            executable.compiled.cost_analysis(),
            executable.compiled.memory_analysis(),
            source="kernel-correction",
            unavailable_reason="Compiler evidence unavailable.",
        )
        cases.append(
            {
                "rows": count,
                "block": args.block,
                "compiler": asdict(evidence),
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "warm": warm.to_seconds_dict(),
                "host_peak_bytes": host_peak,
                "input_bytes": logical_array_bytes((metric, functional)),
                "output_bytes": logical_array_bytes(result),
                "logical_probe_workspace_bytes": 8
                * count
                * (count if args.baseline else args.block),
                "identity_defect": float(np.asarray(result[0])),
                "diagonal_error": float(np.max(np.abs(np.asarray(result[1]) - 1.0))),
                "successful": bool(np.asarray(result[2])),
            }
        )
        jax.clear_caches()
    record = {
        "identity": capture_benchmark_identity(
            Path(__file__).resolve().parents[1],
            Path(__file__).resolve(),
            ("environment", "cases"),
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "route": "baseline" if args.baseline else "blocked",
        "cases": cases,
        "logical_probe_scaling_exponent": fitted_scaling_exponent(
            sizes, [case["logical_probe_workspace_bytes"] for case in cases]
        ),
        "peak_rss_bytes": process_peak_rss_bytes(),
    }
    destination = args.output or Path(__file__).with_name(
        "memory_kernel_correction_baseline.json"
        if args.baseline
        else "memory_kernel_correction.json"
    )
    write_json_atomic(destination, record)
    print(destination)


if __name__ == "__main__":
    main()

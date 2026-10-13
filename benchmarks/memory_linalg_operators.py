"""Selected-block and constraint-certificate workspace scaling (small by default)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.linalg as la
from phydrax.linalg._named_blocks import _SelectedBlockLinearOperator

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


def _measure(action: Callable[[Array], Array], argument: Array) -> dict[str, Any]:
    entry = jax.jit(action)
    compiled, timing = measure_lower_and_compile(
        lambda: entry.lower(argument), lambda lowered: lowered.compile()
    )
    (result, warm), host_peak = host_allocation_peak(
        lambda: measure_repeated(lambda: compiled(argument), warmup=1, repeats=2)
    )
    return {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(
            compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source="jax-compiled-operator-workspace",
            )
        ),
        "output_bytes": logical_array_bytes(result),
        "host_peak_bytes": host_peak,
        "finite": bool(jnp.all(jnp.isfinite(result))),
        "values": np.asarray(result).tolist(),
    }


def _selected(values: Array, workspace: int | None) -> Array:
    size = values.size
    space = la.ArraySpace((size,), dtype=jnp.float64)
    small = la.ArraySpace((4,), dtype=jnp.float64)
    parent = la.FunctionLinearOperator(
        lambda vector: values * jnp.sum(vector),
        source=space,
        target=space,
        operator_id="memory-parent",
    )
    if workspace is None:
        # Original selected-block implementation, retained as a benchmark reference.
        return parent._materialize()[size - 4 :, :4]
    selected = _SelectedBlockLinearOperator(
        parent, ((size - 4, size),), small, ((0, 4),), small
    )
    policy = la.MaterializationPolicy(
        max_entries=16, max_bytes=128, max_workspace_bytes=workspace
    )
    return la.materialize(selected, policy)


def _baseline_norms(matrix: Array, right: Array) -> Array:
    projector = matrix @ right
    return jnp.asarray(
        [
            jnp.linalg.norm(projector @ matrix - matrix),
            jnp.linalg.norm(projector - jnp.eye(matrix.shape[0], dtype=matrix.dtype)),
            jnp.linalg.norm(projector),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[96, 192, 384])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    sizes = [48, 96] if args.smoke else args.sizes
    cases = []
    selected_workspaces: list[int] = []
    certificate_workspaces: list[int] = []
    for size in sizes:
        workspace = None if args.phase == "baseline" else 8 * (2 * size + 4) * 2
        selected = _measure(
            lambda values: _selected(values, workspace),
            jnp.arange(size, dtype=jnp.float64),
        )
        host_matrix = np.random.default_rng(7).normal(size=(size, 8))
        right = jnp.asarray(np.linalg.pinv(host_matrix))
        matrix = jnp.asarray(host_matrix)
        if args.phase == "baseline":
            certificate = _measure(lambda value: _baseline_norms(value, right), matrix)
            certificate_workspace = 8 * (3 * size * size + size * 8)
            selected_workspace = 8 * size * size
        else:
            from phydrax.linalg._constraint_operators import _constraint_projector_norms

            certificate = _measure(
                lambda value: jnp.asarray(
                    _constraint_projector_norms(value, right, False, 4)
                ),
                matrix,
            )
            certificate_workspace = 8 * 4 * (3 * size + 2 * 8)
            selected_workspace = 8 * (2 * size + 4) * 2
        selected["reference_parity"] = bool(
            np.allclose(
                selected["values"],
                np.broadcast_to(
                    np.arange(size - 4, size, dtype=np.float64)[:, None], (4, 4)
                ),
                rtol=1e-12,
                atol=1e-12,
            )
        )
        certificate["reference_parity"] = bool(
            np.allclose(
                certificate["values"],
                [0.0, np.sqrt(size - 8), np.sqrt(8.0)],
                rtol=1e-12,
                atol=1e-10,
            )
        )
        selected_workspaces.append(selected_workspace)
        certificate_workspaces.append(certificate_workspace)
        cases.append(
            {
                "capacity": size,
                "selected": selected,
                "certificate": certificate,
                "selected_logical_workspace_bytes": selected_workspace,
                "certificate_logical_workspace_bytes": certificate_workspace,
            }
        )
    root = Path(__file__).resolve().parents[1]
    record = {
        "phase": args.phase,
        "environment": asdict(capture_environment()),
        "cases": cases,
        "scaling": {
            "selected_logical_workspace_bytes": fitted_scaling_exponent(
                sizes, selected_workspaces
            ),
            "certificate_logical_workspace_bytes": fitted_scaling_exponent(
                sizes, certificate_workspaces
            ),
        },
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "disposition": "unconditional: hidden full materialization and explicit square identity/projector removal",
    }
    record["identity"] = asdict(
        capture_benchmark_identity(root, Path(__file__).resolve(), record.keys())
    )
    suffix = "_baseline" if args.phase == "baseline" else ""
    output = args.output or root / f"benchmarks/memory_linalg_operators{suffix}.json"
    write_json_atomic(output, record)


if __name__ == "__main__":
    main()

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compare unpreconditioned and smoothed-aggregation compatible pressure projection.

Sweeps structured tensor-grid bridges and meshfree positive one-complexes,
recording iterations, host preparation, lowering, compilation, cold and warmed
projection time for each ``CompatiblePressurePreconditioner``. Run with
``--output`` for persistent evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, get_args, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization import (
    CochainDiscretization,
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.discretization.meshfree._exterior import MeshfreeExteriorCalculusPlan
from phydrax.discretization.meshfree._exterior_metric import MeshfreeMetricPolicy
from phydrax.solver import (
    CompatibleIncompressibleProjection,
    CompatiblePressurePreconditioner,
    IncompressibleProjectionResult,
)

from ._io import write_json_atomic
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


def _structured(cells: int, /) -> StructuredCochainBridge:
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(cells) for _ in range(2)), axis_names=("x", "y")
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float64))
    return StructuredCochainBridge(grid)


def _meshfree(side: int, perturbation: float, /) -> CochainDiscretization | None:
    # Radius-1.5 lattice cloud admitted as a native positive one-complex (exact
    # nonnegative edge metric), as in the projection unit fixture. Admission is
    # the metric owner's decision; refused clouds are recorded, not measured.
    rng = np.random.default_rng(3)
    lattice = np.stack(
        np.meshgrid(
            np.arange(side, dtype=np.float64),
            np.arange(side, dtype=np.float64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    points = lattice + perturbation * rng.uniform(-1.0, 1.0, lattice.shape)
    boundary = np.any((lattice == 0.0) | (lattice == side - 1.0), axis=1)
    count = points.shape[0]
    prepared = MeshfreeExteriorCalculusPlan(
        points,
        1.5,
        count * (count - 1) // 2,
        node_volumes=np.ones(count, dtype=np.float64),
        dirichlet=boundary,
        metric_policy=MeshfreeMetricPolicy("nonnegative", tolerance=1e-7),
    ).prepare()
    if not bool(prepared.metric_result.accepted):
        return None
    return prepared.to_cochain()


def _project(
    projection: CompatibleIncompressibleProjection, velocity: Array, /
) -> IncompressibleProjectionResult:
    return projection.project(velocity)


class _ProjectionCompiled(Protocol):
    compiled: jax.stages.Compiled

    def __call__(
        self, projection: CompatibleIncompressibleProjection, velocity: Array, /
    ) -> IncompressibleProjectionResult: ...


class _ProjectionLowered(Protocol):
    def compile(self) -> _ProjectionCompiled: ...


@runtime_checkable
class _ProjectionEntry(Protocol):
    def __call__(
        self, projection: CompatibleIncompressibleProjection, velocity: Array, /
    ) -> IncompressibleProjectionResult: ...

    def lower(
        self, projection: CompatibleIncompressibleProjection, velocity: Array, /
    ) -> _ProjectionLowered: ...


def _require_projection_entry(
    entry: Callable[
        [CompatibleIncompressibleProjection, Array], IncompressibleProjectionResult
    ],
    /,
) -> _ProjectionEntry:
    if not isinstance(entry, _ProjectionEntry):
        raise TypeError(
            "Filtered projection compilation requires a callable lowering entry."
        )
    return entry


_PROJECT = _require_projection_entry(eqx.filter_jit(_project))


def _measure(
    family: str,
    size: int,
    complex: CochainDiscretization | StructuredCochainBridge,
    preconditioner: CompatiblePressurePreconditioner,
    repeats: int,
    /,
) -> dict[str, Any]:
    cochain = complex.cochain if isinstance(complex, StructuredCochainBridge) else complex
    velocity = jnp.sin(jnp.arange(cochain.cell_counts[1], dtype=jnp.float64) / 3.0)
    projection, preparation_seconds = measure_host(
        lambda: CompatibleIncompressibleProjection(complex, preconditioner=preconditioner)
    )
    compiled, timing = measure_lower_and_compile(
        lambda: _PROJECT.lower(projection, velocity),
        lambda lowered: lowered.compile(),
    )
    _, cold_seconds = measure_synchronized(lambda: compiled(projection, velocity))
    result, warm = measure_repeated(
        lambda: compiled(projection, velocity), warmup=1, repeats=repeats
    )
    if not bool(result.successful):
        raise ValueError(
            f"{family} size {size} {preconditioner} projection status {int(result.status)}."
        )
    multigrid = result.multigrid
    return {
        "family": family,
        "size": size,
        "unknowns": cochain.cell_counts[0],
        "edges": cochain.cell_counts[1],
        "preconditioner": preconditioner,
        "status": int(result.status),
        "iterations": int(result.linear.diagnostics.iterations),
        "matvec_count": int(result.linear.diagnostics.matvec_count),
        "relative_residual": float(result.linear.diagnostics.relative_residual),
        "divergence_defect_norm": float(result.divergence_defect_norm),
        "preparation_seconds": preparation_seconds,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "cold_seconds": cold_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(
            compiler_evidence(
                compiled.compiled.cost_analysis(),
                compiled.compiled.memory_analysis(),
                source="jax-compiled-compatible-projection",
            )
        ),
        "retained_preparation_bytes": logical_array_bytes(projection),
        "levels": None if multigrid is None else list(multigrid.level_dimensions),
        "grid_complexity": None if multigrid is None else multigrid.grid_complexity,
        "operator_complexity": None
        if multigrid is None
        else multigrid.operator_complexity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--structured-cells", type=int, nargs="+", default=(16, 32, 64, 128)
    )
    parser.add_argument("--meshfree-sides", type=int, nargs="+", default=(4, 5, 6, 7))
    parser.add_argument("--meshfree-perturbation", type=float, default=0.08)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if any(size < 2 for size in (*arguments.structured_cells, *arguments.meshfree_sides)):
        raise ValueError("Benchmark sizes must be at least two.")
    jax.config.update("jax_enable_x64", True)
    preconditioners: tuple[CompatiblePressurePreconditioner, ...] = get_args(
        CompatiblePressurePreconditioner
    )
    meshfree = [
        (side, _meshfree(side, arguments.meshfree_perturbation))
        for side in arguments.meshfree_sides
    ]
    complexes: list[tuple[str, int, CochainDiscretization | StructuredCochainBridge]] = [
        ("structured", cells, _structured(cells)) for cells in arguments.structured_cells
    ] + [("meshfree", side, cochain) for side, cochain in meshfree if cochain is not None]
    rows = [
        _measure(family, size, complex, preconditioner, arguments.repeats)
        for family, size, complex in complexes
        for preconditioner in preconditioners
    ]
    for row in rows:
        print(
            f"{row['family']:>10} {row['unknowns']:>6} {row['preconditioner']:>20} "
            f"iters={row['iterations']:>4} setup={row['preparation_seconds']:.2f}s "
            f"compile={row['compilation_seconds']:.2f}s "
            f"warm={row['warm']['median_seconds']:.4f}s levels={row['levels']}",
            flush=True,
        )
    root = Path(__file__).resolve().parent.parent
    identity = capture_benchmark_identity(root, Path(__file__), rows[0])
    record = {
        "benchmark_identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "meshfree_perturbation": arguments.meshfree_perturbation,
        "meshfree_metric_refused_sides": [
            side for side, cochain in meshfree if cochain is None
        ],
        "cases": rows,
    }
    if arguments.output is not None:
        write_json_atomic(arguments.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

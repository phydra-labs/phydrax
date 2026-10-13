"""Compiler workspace gates for exact ML pair reductions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

import jax
import jax.numpy as jnp

from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
    process_peak_rss_bytes,
)
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.linalg.eigen import RestartedLanczos
from phydrax.ml._numerics._pairwise import (
    blocked_logsumexp,
    blocked_top_k,
    pair_block_size,
    pairwise_distances,
)
from phydrax.ml.clustering._common import pairwise_distances as clustering_distances
from phydrax.ml.clustering._density import _density_one
from phydrax.ml.clustering._spectral import _partial_spectrum


class MemoryRecord(TypedDict):
    name: str
    capacity: int
    temporary_bytes: int
    output_bytes: int | None
    lowering_seconds: float
    compilation_seconds: float
    generated_code_bytes: int | None
    logical_workspace_bytes: int
    warmed_seconds: list[float] | None
    compiler_host_peak_bytes: int
    input_bytes: int
    reference_warmed_seconds: list[float] | None
    workspace_budget_bytes: int
    pair_tile_extent: int


def _dense_density(x: jax.Array, w: jax.Array) -> tuple[jax.Array, ...]:
    """Untouched DBSCAN baseline, retained only as an independent benchmark route."""
    n = x.shape[0]
    active = w > 0.0
    adjacency = (
        (clustering_distances(x, "euclidean") <= 0.3) & active[:, None] & active[None, :]
    )
    core = active & (adjacency @ w >= 3.0)
    root = jnp.where(core, jnp.arange(n, dtype=jnp.int32), n)
    core_edges = adjacency & core[:, None] & core[None, :]

    def propagate(_: jax.Array, labels: jax.Array) -> jax.Array:
        candidate = jnp.where(core_edges, labels[None, :], n)
        return jnp.where(core, jnp.minimum(labels, jnp.min(candidate, axis=-1)), n)

    root = jax.lax.fori_loop(0, n, propagate, root)
    adjacent_root = jnp.min(
        jnp.where(adjacency & core[None, :], root[None, :], n), axis=-1
    )
    root = jnp.where(
        core, root, jnp.where(active & (adjacent_root < n), adjacent_root, n)
    )
    is_root = core & (root == jnp.arange(n, dtype=jnp.int32))
    cluster_count = jnp.sum(is_root)
    cluster_label = jnp.sum(
        is_root[None, :] & (jnp.arange(n)[None, :] < root[:, None]), axis=-1
    ).astype(jnp.int32)
    labels = jnp.where(root < n, cluster_label, -1)
    within_capacity = labels < 4
    labels = jnp.where(within_capacity, labels, -1)
    membership = (
        jax.nn.one_hot(jnp.maximum(labels, 0), 4, dtype=w.dtype)
        * ((labels >= 0) * w)[:, None]
    )
    mass = membership.sum(axis=0)
    centers = membership.T @ x / jnp.maximum(mass, jnp.finfo(w.dtype).tiny)[:, None]
    order = jnp.argsort(jnp.where(core & within_capacity, jnp.arange(n), n))[:n]
    core_active = (order < n) & core[order] & within_capacity[order]
    safe_order = jnp.minimum(order, n - 1)
    return (
        centers,
        mass,
        labels,
        x[safe_order],
        jnp.where(core_active, labels[safe_order], -1),
        core_active,
        cluster_count,
        jnp.sum(core),
        jnp.sum(active),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = [256, 512] if args.smoke else [256, 512, 1024]
    records: list[MemoryRecord] = []
    for n in sizes:
        x = jnp.sin(jnp.arange(n * 3, dtype=jnp.float64).reshape(n, 3))
        w = jnp.ones(n, dtype=jnp.float64)
        for name in ("knn", "kde", "dbscan"):
            extra_row_bytes = 192 if name == "knn" else 0
            tile = (
                n
                if args.baseline
                else pair_block_size(
                    n, n, 3, 8, 4 * 1024 * 1024, extra_row_bytes=extra_row_bytes
                )
            )

            def kernel(x: jax.Array, w: jax.Array) -> object:
                if name == "knn":
                    if not args.baseline:
                        return blocked_top_k(x, x, w > 0, 4)
                    return jax.lax.top_k(-pairwise_distances(x, x), 4)
                if name == "kde":
                    if not args.baseline:
                        return blocked_logsumexp(
                            x, x, jnp.zeros_like(w), jnp.asarray(1.0)
                        )
                    return jax.scipy.special.logsumexp(
                        -pairwise_distances(x, x, metric="squared-euclidean"), axis=-1
                    )
                if args.baseline:
                    return _dense_density(x, w)
                return _density_one(x, w, 0.3, 3.0, 4, n)

            (compiled, timing), host_peak = host_allocation_peak(
                lambda: measure_lower_and_compile(
                    lambda: jax.jit(kernel).lower(x, w), lambda value: value.compile()
                )
            )
            warmed = None
            if n == sizes[0] and args.baseline:
                _, samples = measure_repeated(lambda: compiled(x, w), warmup=2, repeats=7)
                warmed = list(samples.samples_seconds)
            reference_warmed = None
            if n == sizes[0] and not args.baseline:

                def reference(x: jax.Array, w: jax.Array) -> object:
                    if name == "knn":
                        return jax.lax.top_k(-pairwise_distances(x, x), 4)
                    if name == "kde":
                        return jax.scipy.special.logsumexp(
                            -pairwise_distances(x, x, metric="squared-euclidean"), axis=-1
                        )
                    return _dense_density(x, w)

                reference_compiled = jax.jit(reference).lower(x, w).compile()
                candidate_times: list[float] = []
                reference_times: list[float] = []

                def paired_run() -> tuple[object, object]:
                    candidate, candidate_time = measure_synchronized(
                        lambda: compiled(x, w)
                    )
                    reference_value, reference_time = measure_synchronized(
                        lambda: reference_compiled(x, w)
                    )
                    candidate_times.append(candidate_time)
                    reference_times.append(reference_time)
                    return candidate, reference_value

                measure_repeated(paired_run, warmup=2, repeats=7)
                warmed = candidate_times[2:]
                reference_warmed = reference_times[2:]
            evidence = compiler_evidence(
                compiled.cost_analysis(), compiled.memory_analysis(), source="jax"
            )
            if evidence.temporary_bytes is None:
                raise RuntimeError("Compiler temporary-memory evidence is required.")
            records.append(
                {
                    "name": name,
                    "capacity": n,
                    "temporary_bytes": evidence.temporary_bytes,
                    "output_bytes": evidence.output_bytes,
                    "lowering_seconds": timing.lowering_seconds,
                    "compilation_seconds": timing.compilation_seconds,
                    "generated_code_bytes": evidence.generated_code_bytes,
                    "logical_workspace_bytes": 8 * 6 * tile * tile
                    + tile * extra_row_bytes,
                    "workspace_budget_bytes": 4 * 1024 * 1024,
                    "pair_tile_extent": tile,
                    "input_bytes": logical_array_bytes((x, w)),
                    "warmed_seconds": warmed,
                    "reference_warmed_seconds": reference_warmed,
                    "compiler_host_peak_bytes": host_peak,
                }
            )
    if not args.baseline:
        for n in sizes:
            x = jnp.sin(jnp.arange(n * 3, dtype=jnp.float64).reshape(n, 3))
            w = jnp.ones(n, dtype=jnp.float64)

            def spectrum(x: jax.Array, w: jax.Array) -> tuple[jax.Array, ...]:
                return _partial_spectrum(x, w, jnp.asarray(0.5), 3, RestartedLanczos())

            (compiled, timing), host_peak = host_allocation_peak(
                lambda: measure_lower_and_compile(
                    lambda: jax.jit(spectrum).lower(x, w), lambda value: value.compile()
                )
            )
            evidence = compiler_evidence(
                compiled.cost_analysis(), compiled.memory_analysis(), source="jax"
            )
            if evidence.temporary_bytes is None:
                raise RuntimeError("Compiler temporary-memory evidence is required.")
            warmed = None
            if n == sizes[0]:
                _, samples = measure_repeated(lambda: compiled(x, w), warmup=1, repeats=2)
                warmed = list(samples.samples_seconds)
            records.append(
                {
                    "name": "spectral",
                    "capacity": n,
                    "temporary_bytes": evidence.temporary_bytes,
                    "output_bytes": evidence.output_bytes,
                    "generated_code_bytes": evidence.generated_code_bytes,
                    "logical_workspace_bytes": 8 * (n * 14 + 64 * 64 * 3),
                    "workspace_budget_bytes": 512 * 1024 * 1024,
                    "pair_tile_extent": 64,
                    "input_bytes": logical_array_bytes((x, w)),
                    "warmed_seconds": warmed,
                    "compiler_host_peak_bytes": host_peak,
                    "reference_warmed_seconds": None,
                    "lowering_seconds": timing.lowering_seconds,
                    "compilation_seconds": timing.compilation_seconds,
                }
            )
    root = Path(__file__).resolve().parents[1]
    result = {
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root, Path(__file__), ("temporary_bytes", "output_bytes")
        ).to_dict(),
        "records": records,
        "rss_bytes": process_peak_rss_bytes(),
        "timing_acceptance": "deferred to a quiet serialized wave",
    }
    if not args.baseline:
        import json

        baseline = json.loads(
            (root / "benchmarks/memory_ml_neighbors_baseline.json").read_text()
        )
        result["gates"] = {
            name: evaluate_memory_gate(
                sizes,
                [
                    r["temporary_bytes"]
                    for r in baseline["records"]
                    if r["name"] == name and r["capacity"] in sizes
                ],
                [r["temporary_bytes"] for r in records if r["name"] == name],
                MemoryGatePolicy(),
            ).to_dict()
            for name in ("knn", "kde", "dbscan")
        }
        result["exponents"] = {
            name: fitted_scaling_exponent(
                sizes,
                [max(r["temporary_bytes"], 1) for r in records if r["name"] == name],
            )
            for name in ("knn", "kde", "dbscan")
        }
    write_json_atomic(
        args.output
        if args.output is not None
        else root
        / (
            "benchmarks/memory_ml_neighbors_baseline.json"
            if args.baseline
            else "benchmarks/memory_ml_neighbors.json"
        ),
        result,
    )


if __name__ == "__main__":
    main()

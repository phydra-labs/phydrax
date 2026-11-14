"""Phase-separated cost of k-means fits, EMA mini-batch fits, and assignment.

Every surface is lowered and compiled separately from warmed execution. Compiler
temporaries are recorded over a sample/cluster/feature grid; warmed timings are
recorded on the subset that also fits the unbounded baseline. Gates compare the
candidate against the committed baseline record, measured on the same grid and
machine from dev 7bdc501d0: Lloyd and prediction over complete
sample-by-center differences, and the with-replacement Sculley mini-batch
(``batch_size=32``, 64 iterations) that EMA mini-batch k-means replaced.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import jax
import jax.numpy as jnp

from benchmarks._io import write_json_atomic
from benchmarks._memory import evaluate_memory_gate, MemoryGatePolicy
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax.ml import MLBatch
from phydrax.ml.clustering import HardClusterModel, KMeans, MiniBatchKMeans


_MEBIBYTE = 1024 * 1024
_KMEANS_ITERATIONS = 8
_EPOCHS = 2.0
_SAMPLES = (10_000, 100_000, 1_000_000)
_CLUSTERS = (16, 256, 4096)
_FEATURES = (2, 16, 96)
_TIMED = frozenset(
    (n, k, d) for n in (10_000, 100_000) for k in (16, 256) for d in (2, 16)
)
# The first workspace is the recipe default the grid and gates use.
_WORKSPACES = (64 * _MEBIBYTE, 4 * _MEBIBYTE, 256 * _MEBIBYTE)
_WORKSPACE_CASES = ((100_000, 256, 16), (1_000_000, 4096, 96))
_COMPILE_EPOCHS = (2.0, 20.0)
_TIME_TOLERANCE = 0.10
_COMPILE_TOLERANCE = 0.20


class Record(TypedDict):
    surface: str
    samples: int
    clusters: int
    features: int
    workspace_bytes: int | None
    epochs: float | None
    lowering_seconds: float
    compilation_seconds: float
    temporary_bytes: int | None
    output_bytes: int | None
    generated_code_bytes: int | None
    retained_bytes: int | None
    warmed: dict[str, Any] | None
    iterations: int | None
    warmed_seconds_per_iteration: float | None


def _data(n: int, d: int) -> jax.Array:
    return jax.random.normal(jax.random.key(1), (n, d), dtype=jnp.float64)


def _measure(
    surface: str,
    function: Callable[..., jax.Array],
    arguments: tuple[jax.ShapeDtypeStruct, ...],
    concrete: Callable[[], tuple[jax.Array, ...]] | None,
    shape: tuple[int, int, int],
    *,
    workspace_bytes: int | None,
    epochs: float | None = None,
    iterations: int | None = None,
) -> Record:
    jitted = jax.jit(function)
    compiled, timing = measure_lower_and_compile(
        lambda: jitted.lower(*arguments), lambda lowered: lowered.compile()
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(), compiled.memory_analysis(), source="xla"
    )
    warmed = None
    per_iteration = None
    retained = None
    if concrete is not None:
        values = concrete()
        result, distribution = measure_repeated(
            lambda: compiled(*values), warmup=1, repeats=5
        )
        retained = logical_array_bytes(result)
        warmed = distribution.to_seconds_dict()
        median = distribution.median_seconds
        if iterations is not None and median is not None:
            per_iteration = median / iterations
    n, k, d = shape
    record: Record = {
        "surface": surface,
        "samples": n,
        "clusters": k,
        "features": d,
        "workspace_bytes": workspace_bytes,
        "epochs": epochs,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "temporary_bytes": evidence.temporary_bytes,
        "output_bytes": evidence.output_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
        "retained_bytes": retained,
        "warmed": warmed,
        "iterations": iterations,
        "warmed_seconds_per_iteration": per_iteration,
    }
    print(
        surface,
        shape,
        workspace_bytes,
        epochs,
        evidence.temporary_bytes,
        per_iteration
        if per_iteration is not None
        else warmed and warmed["median_seconds"],
        flush=True,
    )
    return record


def _kmeans(shape: tuple[int, int, int], workspace: int, timed: bool) -> Record:
    n, k, d = shape
    recipe = KMeans(
        k,
        max_iterations=_KMEANS_ITERATIONS,
        tolerance=0.0,
        initialization="first",
        workspace_bytes=workspace,
    )
    return _measure(
        "k-means-fit",
        lambda x: recipe.fit_batch(MLBatch(x)).diagnostics.cluster_mass,
        (jax.ShapeDtypeStruct((n, d), jnp.float64),),
        (lambda: (_data(n, d),)) if timed else None,
        shape,
        workspace_bytes=workspace,
        iterations=_KMEANS_ITERATIONS,
    )


def _minibatch(
    shape: tuple[int, int, int], workspace: int, epochs: float, timed: bool
) -> Record:
    n, k, d = shape
    recipe = MiniBatchKMeans(k, epochs=epochs, workspace_bytes=workspace)
    return _measure(
        "mini-batch-k-means-fit",
        lambda x: (
            recipe.fit_batch(MLBatch(x), key=jax.random.key(0)).diagnostics.cluster_mass
        ),
        (jax.ShapeDtypeStruct((n, d), jnp.float64),),
        (lambda: (_data(n, d),)) if timed else None,
        shape,
        workspace_bytes=workspace,
        epochs=epochs,
    )


def _prediction(shape: tuple[int, int, int], timed: bool) -> Record:
    n, k, d = shape

    def predict(x: jax.Array, centers: jax.Array) -> jax.Array:
        return HardClusterModel(
            centers, jnp.ones((k,), dtype=jnp.bool_), method="benchmark"
        )(x)

    return _measure(
        "hard-cluster-prediction",
        predict,
        (
            jax.ShapeDtypeStruct((n, d), jnp.float64),
            jax.ShapeDtypeStruct((k, d), jnp.float64),
        ),
        (lambda: (_data(n, d), jax.random.normal(jax.random.key(2), (k, d))))
        if timed
        else None,
        shape,
        workspace_bytes=None,
    )


def _find(
    records: list[dict[str, Any]],
    surface: str,
    shape: tuple[int, int, int],
    *,
    workspace_bytes: int | None = None,
    epochs: float | None = None,
) -> dict[str, Any]:
    n, k, d = shape
    for record in records:
        if (
            record["surface"] == surface
            and (record["samples"], record["clusters"], record["features"]) == (n, k, d)
            and (workspace_bytes is None or record["workspace_bytes"] == workspace_bytes)
            and (epochs is None or record.get("epochs") == epochs)
        ):
            return record
    raise ValueError(f"Missing {surface} record for {shape}.")


def _memory_gates(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for surface in ("k-means-fit", "mini-batch-k-means-fit", "hard-cluster-prediction"):
        workspace = None if surface == "hard-cluster-prediction" else _WORKSPACES[0]
        epochs = _EPOCHS if surface == "mini-batch-k-means-fit" else None
        shape_baseline = [
            _find(baseline, surface, (n, 4096, 96))["temporary_bytes"] for n in _SAMPLES
        ]
        shape_candidate = [
            _find(
                candidate,
                surface,
                (n, 4096, 96),
                workspace_bytes=workspace,
                epochs=epochs,
            )["temporary_bytes"]
            for n in _SAMPLES
        ]
        decision = evaluate_memory_gate(
            _SAMPLES, shape_baseline, shape_candidate, MemoryGatePolicy()
        ).to_dict()
        by_cluster = [
            _find(
                candidate,
                surface,
                (_SAMPLES[-1], k, 96),
                workspace_bytes=workspace,
                epochs=epochs,
            )["temporary_bytes"]
            for k in _CLUSTERS
        ]
        gates[surface] = {
            "largest_sample_gate": decision,
            "temporary_bytes_by_cluster_count": dict(zip(_CLUSTERS, by_cluster)),
            # Workspace-bounded tiles must not grow with the cluster count.
            "cluster_count_bounded": by_cluster[-1] <= 2 * by_cluster[0],
        }
    return gates


def _time_gate(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    for k in (16, 256):
        for d in (2, 16):
            shape = (100_000, k, d)
            reference = _find(baseline, "k-means-fit", shape)[
                "warmed_seconds_per_iteration"
            ]
            measured = _find(
                candidate, "k-means-fit", shape, workspace_bytes=_WORKSPACES[0]
            )["warmed_seconds_per_iteration"]
            rows.append(
                {
                    "shape": list(shape),
                    "baseline_seconds_per_iteration": reference,
                    "candidate_seconds_per_iteration": measured,
                    "ratio": measured / reference,
                    "passed": measured <= (1.0 + _TIME_TOLERANCE) * reference,
                }
            )
    return {"tolerance": _TIME_TOLERANCE, "rows": rows}


def _compile_seconds(epochs: float, repeats: int) -> list[float]:
    """Lower and compile a fresh jitted fit each time, so no cache is reused."""
    n, k, d = _WORKSPACE_CASES[0]
    spec = jax.ShapeDtypeStruct((n, d), jnp.float64)
    seconds = []
    for _ in range(repeats):
        recipe = MiniBatchKMeans(k, epochs=epochs, workspace_bytes=_WORKSPACES[0])
        jitted = jax.jit(
            lambda x: (
                recipe.fit_batch(
                    MLBatch(x), key=jax.random.key(0)
                ).diagnostics.cluster_mass
            )
        )
        _, timing = measure_lower_and_compile(
            lambda: jitted.lower(spec), lambda lowered: lowered.compile()
        )
        seconds.append(timing.compilation_seconds)
    return seconds


def _compile_gate() -> dict[str, Any]:
    """Compile cost must not grow with the number of passes."""
    samples = {str(epochs): _compile_seconds(epochs, 5) for epochs in _COMPILE_EPOCHS}
    short, long = (statistics.median(samples[str(epochs)]) for epochs in _COMPILE_EPOCHS)
    return {
        "epochs": list(_COMPILE_EPOCHS),
        "compilation_seconds": samples,
        "median_compilation_seconds": [short, long],
        "passed": abs(long - short) <= _COMPILE_TOLERANCE * short,
    }


def run() -> list[Record]:
    records: list[Record] = []
    for n in _SAMPLES:
        for k in _CLUSTERS:
            for d in _FEATURES:
                shape = (n, k, d)
                timed = shape in _TIMED
                records.append(_kmeans(shape, _WORKSPACES[0], timed))
                records.append(_minibatch(shape, _WORKSPACES[0], _EPOCHS, timed))
                records.append(_prediction(shape, timed))
    for shape in _WORKSPACE_CASES:
        timed = shape in _TIMED or shape == _WORKSPACE_CASES[0]
        for workspace in _WORKSPACES[1:]:
            records.append(_kmeans(shape, workspace, timed))
            records.append(_minibatch(shape, workspace, _EPOCHS, timed))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/clustering_minibatch_performance_baseline.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/clustering_minibatch_performance.json"),
    )
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["records"]
    candidate = [dict(record) for record in run()]
    payload = {
        "environment": capture_environment().to_dict(),
        "records": candidate,
        "gates": {
            "memory": _memory_gates(baseline, candidate),
            "k_means_time": _time_gate(baseline, candidate),
            "mini_batch_compile": _compile_gate(),
        },
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

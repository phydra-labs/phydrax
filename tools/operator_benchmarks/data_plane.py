from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from statistics import median

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_lower_and_compile, measure_synchronized
from phydrax._data_plane import StatelessIndexPermutation


@dataclass(frozen=True)
class OperatorDataPlaneBenchmark:
    cases: int
    batch_size: int
    resolution: int
    prefetch: int
    read_latency_seconds: float
    consumer_latency_seconds: float
    ordering_lowering_seconds: float
    ordering_compilation_seconds: float
    ordering_first_execution_seconds: float
    fingerprint_cold_seconds: float
    fingerprint_warm_seconds: float
    fingerprint_case_reads: int
    sync_first_batch_seconds: float
    prefetched_first_batch_seconds: float
    sync_epoch_seconds: float
    prefetched_epoch_seconds: float
    sync_cases_per_second: float
    prefetched_cases_per_second: float
    sync_peak_host_bytes: int
    prefetched_peak_host_bytes: int
    current_batch_device_bytes: int
    peak_device_bytes: int | None
    resume_gate_seconds: float
    resume_gate_case_reads: int
    exact_order_match: bool

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


def _dataset(cases: int, resolution: int):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, resolution),
        quadrature_weights=jnp.full((resolution,), 1.0 / resolution),
    )
    values = jnp.arange(cases, dtype=jnp.float32)[:, None] + axis.nodes[None, :]
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"solution": 2.0 * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )


def _callback_source(dataset, *, read_latency: float, fingerprint: str | None = None):
    backing = phx.nn.operator.InMemoryOperatorCaseSource(dataset)
    reads: list[int] = []

    def metadata_reader(index):
        return backing.case_metadata(index)

    def case_reader(index, request):
        reads.append(index)
        if read_latency:
            time.sleep(read_latency)
        return backing.read_case(index, request=request)

    source = phx.nn.operator.CallbackOperatorCaseSource(
        dataset.size,
        metadata_reader=metadata_reader,
        case_reader=case_reader,
        content_fingerprint=(
            backing.content_fingerprint if fingerprint is None else fingerprint
        ),
        background_read_safe=True,
        configuration={"adapter": "operator-data-plane-benchmark", "revision": 1},
    )
    return source, reads


def _device_bytes(batch) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves((batch.batch, batch.targets))
        if isinstance(leaf, jax.Array)
    )


def _peak_device_bytes() -> int | None:
    statistics = jax.devices()[0].memory_stats()
    if statistics is None:
        return None
    for key in ("peak_bytes_in_use", "peak_bytes_in_use.max"):
        if key in statistics:
            return int(statistics[key])
    return None


def _profile_epoch(loader, *, consumer_latency: float):
    tracemalloc.start()
    started = time.perf_counter()
    first_elapsed = None
    order = []
    current_device_bytes = 0
    with loader.epoch(0) as batches:
        for batch in batches:
            jax.block_until_ready((batch.batch, batch.targets))
            if first_elapsed is None:
                first_elapsed = time.perf_counter() - started
                current_device_bytes = _device_bytes(batch)
            order.extend(batch.indices)
            if consumer_latency:
                time.sleep(consumer_latency)
    elapsed = time.perf_counter() - started
    _, peak_host_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if first_elapsed is None:
        raise ValueError("Benchmark loader emitted no batches.")
    return (
        first_elapsed,
        elapsed,
        int(peak_host_bytes),
        tuple(order),
        current_device_bytes,
    )


def run_data_plane_benchmark(
    *,
    cases: int = 4096,
    batch_size: int = 64,
    resolution: int = 128,
    prefetch: int = 2,
    read_latency_seconds: float = 0.0,
    consumer_latency_seconds: float = 0.0,
    repetitions: int = 3,
) -> OperatorDataPlaneBenchmark:
    """Measure loader-only compilation, identity, latency, throughput, and memory."""
    if int(cases) <= 0 or int(batch_size) <= 0 or int(resolution) <= 0:
        raise ValueError("cases, batch_size, and resolution must be positive.")
    if int(prefetch) <= 0:
        raise ValueError("prefetch must be positive for comparative measurement.")
    if float(read_latency_seconds) < 0.0 or float(consumer_latency_seconds) < 0.0:
        raise ValueError("read and consumer latencies must be nonnegative.")
    if int(repetitions) <= 0:
        raise ValueError("repetitions must be positive.")

    dataset = _dataset(int(cases), int(resolution))
    positions = jnp.arange(int(cases), dtype=jnp.int32)
    permutation = StatelessIndexPermutation(int(cases), 1729, 0)
    compiled_order, ordering_compilation = measure_lower_and_compile(
        lambda: jax.jit(jax.vmap(permutation.jax)).lower(positions),
        lambda lowered: lowered.compile(),
    )
    _, ordering_first_execution_seconds = measure_synchronized(
        lambda: compiled_order(positions)
    )

    fingerprint_source = phx.nn.operator.InMemoryOperatorCaseSource(dataset)
    fingerprint_started = time.perf_counter()
    content_fingerprint = fingerprint_source.content_fingerprint
    fingerprint_cold_seconds = time.perf_counter() - fingerprint_started
    fingerprint_started = time.perf_counter()
    assert fingerprint_source.content_fingerprint == content_fingerprint
    fingerprint_warm_seconds = time.perf_counter() - fingerprint_started

    fingerprint_callback, fingerprint_reads = _callback_source(
        dataset,
        read_latency=float(read_latency_seconds),
    )
    fingerprint_loader = phx.nn.operator.training.OperatorBatchLoader(
        fingerprint_callback,
        batch_size=int(batch_size),
        seed=1729,
        prefetch=int(prefetch),
    )
    expected_fingerprint = fingerprint_loader.fingerprint

    sync_profiles = []
    prefetched_profiles = []
    for _ in range(int(repetitions)):
        sync_source, _ = _callback_source(
            dataset,
            read_latency=float(read_latency_seconds),
        )
        prefetched_source, _ = _callback_source(
            dataset,
            read_latency=float(read_latency_seconds),
        )
        sync_profiles.append(
            _profile_epoch(
                phx.nn.operator.training.OperatorBatchLoader(
                    sync_source,
                    batch_size=int(batch_size),
                    seed=1729,
                    prefetch=0,
                ),
                consumer_latency=float(consumer_latency_seconds),
            )
        )
        prefetched_profiles.append(
            _profile_epoch(
                phx.nn.operator.training.OperatorBatchLoader(
                    prefetched_source,
                    batch_size=int(batch_size),
                    seed=1729,
                    prefetch=int(prefetch),
                ),
                consumer_latency=float(consumer_latency_seconds),
            )
        )

    sync_first = median(profile[0] for profile in sync_profiles)
    prefetched_first = median(profile[0] for profile in prefetched_profiles)
    sync_epoch = median(profile[1] for profile in sync_profiles)
    prefetched_epoch = median(profile[1] for profile in prefetched_profiles)
    sync_peak = int(median(profile[2] for profile in sync_profiles))
    prefetched_peak = int(median(profile[2] for profile in prefetched_profiles))
    exact_order_match = all(
        sync[3] == prefetched[3]
        for sync, prefetched in zip(sync_profiles, prefetched_profiles, strict=True)
    )

    incompatible_source, resume_gate_reads = _callback_source(
        dataset,
        read_latency=float(read_latency_seconds),
        fingerprint="benchmark:incompatible-content",
    )
    incompatible_loader = phx.nn.operator.training.OperatorBatchLoader(
        incompatible_source,
        batch_size=int(batch_size),
        seed=1729,
        prefetch=int(prefetch),
    )
    gate_started = time.perf_counter()
    mismatch = incompatible_loader.fingerprint != expected_fingerprint
    resume_gate_seconds = time.perf_counter() - gate_started
    if not mismatch:
        raise AssertionError("Benchmark resume gate failed to detect changed content.")

    return OperatorDataPlaneBenchmark(
        cases=int(cases),
        batch_size=int(batch_size),
        resolution=int(resolution),
        prefetch=int(prefetch),
        consumer_latency_seconds=float(consumer_latency_seconds),
        read_latency_seconds=float(read_latency_seconds),
        ordering_lowering_seconds=ordering_compilation.lowering_seconds,
        ordering_compilation_seconds=ordering_compilation.compilation_seconds,
        ordering_first_execution_seconds=ordering_first_execution_seconds,
        fingerprint_cold_seconds=fingerprint_cold_seconds,
        fingerprint_warm_seconds=fingerprint_warm_seconds,
        fingerprint_case_reads=len(fingerprint_reads),
        sync_first_batch_seconds=sync_first,
        prefetched_first_batch_seconds=prefetched_first,
        sync_epoch_seconds=sync_epoch,
        prefetched_epoch_seconds=prefetched_epoch,
        sync_cases_per_second=int(cases) / sync_epoch,
        prefetched_cases_per_second=int(cases) / prefetched_epoch,
        sync_peak_host_bytes=sync_peak,
        prefetched_peak_host_bytes=prefetched_peak,
        current_batch_device_bytes=sync_profiles[0][4],
        peak_device_bytes=_peak_device_bytes(),
        resume_gate_seconds=resume_gate_seconds,
        resume_gate_case_reads=len(resume_gate_reads),
        exact_order_match=exact_order_match,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the operator data plane.")
    parser.add_argument("--cases", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--read-latency-seconds", type=float, default=0.0)
    parser.add_argument("--consumer-latency-seconds", type=float, default=0.0)
    parser.add_argument("--repetitions", type=int, default=3)
    arguments = parser.parse_args()
    result = run_data_plane_benchmark(
        cases=arguments.cases,
        batch_size=arguments.batch_size,
        resolution=arguments.resolution,
        prefetch=arguments.prefetch,
        read_latency_seconds=arguments.read_latency_seconds,
        repetitions=arguments.repetitions,
        consumer_latency_seconds=arguments.consumer_latency_seconds,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()


__all__ = ["OperatorDataPlaneBenchmark", "run_data_plane_benchmark"]

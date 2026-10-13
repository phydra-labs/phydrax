"""Measure process projection and local-instrument contraction against Kronecker references."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.tensor_network import (
    CausalProcessTensor,
    CombLegSpec,
    ProcessMemoryProjectionResult,
    project_process_memory_subspace,
    QuantumInstrument,
)

from ._io import write_json_atomic
from ._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
    process_peak_rss_bytes,
)
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def _selector_projection(process: CausalProcessTensor) -> ProcessMemoryProjectionResult:
    system, old = process.spec.system_dimension, process.spec.memory_dimension
    retained = old // 2
    selector = jnp.kron(
        jnp.eye(system, dtype=process.initial_state.dtype),
        jnp.eye(old, dtype=process.initial_state.dtype)[:retained],
    )
    projector = jnp.conj(selector.T) @ selector
    discarded = jnp.eye(system * old, dtype=process.initial_state.dtype) - projector
    leakage = jnp.max(
        jnp.stack(
            [
                jnp.sqrt(jnp.sum(jnp.abs(discarded @ operator @ projector) ** 2))
                for values in process.channel_kraus
                for operator in values
            ]
        )
    )
    initial = selector @ process.initial_state @ jnp.conj(selector.T)
    weight = jnp.real(jnp.trace(initial))
    channels = tuple(
        jnp.stack([selector @ operator @ jnp.conj(selector.T) for operator in values])
        for values in process.channel_kraus
    )
    compressed = CausalProcessTensor(
        CombLegSpec(
            system,
            retained,
            process.spec.slot_count,
            spec_id=f"{process.spec.spec_id}:memory-{retained}",
        ),
        initial / jnp.where(weight > 0, weight, 1.0),
        channels,
        process_id=f"{process.process_id}:memory-{retained}",
    )
    return ProcessMemoryProjectionResult(
        compressed,
        1.0 - weight,
        jnp.max(compressed.channel_completeness_residuals),
        weight,
        leakage,
    )


def _kron_contract(
    process: CausalProcessTensor, instrument: QuantumInstrument
) -> jax.Array:
    state = process.initial_state
    memory = process.spec.memory_dimension
    updated = jnp.zeros_like(state)
    for operator in instrument.kraus[0]:
        local = jnp.kron(operator, jnp.eye(memory, dtype=state.dtype))
        updated = updated + local @ state @ jnp.conj(local.T)
    state = sum(
        operator @ updated @ jnp.conj(operator.T) for operator in process.channel_kraus[0]
    )
    probability = jnp.real(jnp.trace(state))
    reduced = jnp.trace(state.reshape(2, memory, 2, memory), axis1=1, axis2=3)
    return reduced / probability


class _Compiled(Protocol):
    compiled: jax.stages.Compiled

    def __call__(self, process: CausalProcessTensor, /) -> Any: ...


class _Lowered(Protocol):
    def compile(self) -> _Compiled: ...


@runtime_checkable
class _Entry(Protocol):
    def __call__(self, process: CausalProcessTensor, /) -> Any: ...

    def lower(self, process: CausalProcessTensor, /) -> _Lowered: ...


def _measure(
    operation: Callable[[CausalProcessTensor], Any],
    process: CausalProcessTensor,
    repeats: int,
) -> dict[str, Any]:
    entry = eqx.filter_jit(operation)
    if not isinstance(entry, _Entry):
        raise TypeError("Process compilation requires a callable lowering entry.")
    executable, timing = measure_lower_and_compile(
        lambda: entry.lower(process), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(
        lambda: executable(process), warmup=1, repeats=repeats
    )
    evidence = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="jax-compiled-process-memory",
        unavailable_reason="Compiler estimates unavailable.",
    )
    return {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "output_bytes": logical_array_bytes(result),
    }


def _case(memory: int, repeats: int) -> list[dict[str, Any]]:
    dimension = 2 * memory
    initial = jnp.eye(dimension, dtype=jnp.complex128) / dimension
    # Two nontrivial Kraus terms vary both memory and local Kraus capacity.
    channel = jnp.stack(
        (
            jnp.eye(dimension, dtype=jnp.complex128) * jnp.sqrt(0.7),
            jnp.diag(jnp.exp(1j * jnp.arange(dimension, dtype=jnp.float64)))
            * jnp.sqrt(0.3),
        )
    )
    process = CausalProcessTensor(
        CombLegSpec(2, memory, 1), initial, (channel,), process_id="memory-benchmark"
    )
    instrument = QuantumInstrument(
        jnp.stack(
            (
                jnp.eye(2, dtype=jnp.complex128) * jnp.sqrt(0.6),
                jnp.asarray([[0, 1j], [1, 0]], dtype=jnp.complex128) * jnp.sqrt(0.4),
            )
        )[None],
        jnp.asarray([True]),
        jnp.asarray([[True, True]]),
        instrument_id="memory-benchmark-local",
    )
    # Outcome selection is host metadata in contract(); keep that guard host-side.
    instrument = eqx.tree_at(
        lambda value: value.outcome_active,
        instrument,
        np.asarray([True], dtype=np.bool_),
    )
    operations = (
        ("projection", "selector", _selector_projection),
        (
            "projection",
            "production",
            lambda value: project_process_memory_subspace(value, memory // 2),
        ),
        ("contract", "kronecker", lambda value: _kron_contract(value, instrument)),
        (
            "contract",
            "production",
            lambda value: value.contract((instrument,), (0,)).final_system_state,
        ),
    )
    rows: list[dict[str, Any]] = []
    for family, route, operation in operations:
        row = _measure(operation, process, repeats)
        row.update(
            family=family,
            route=route,
            capacity=memory,
            retained_input_bytes=logical_array_bytes(process),
            logical_selector_bytes=(dimension * dimension + memory * dimension) * 16
            if route == "selector"
            else 0,
            logical_expanded_local_bytes=dimension * dimension * 16
            if route == "kronecker"
            else 0,
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    args = parser.parse_args()
    sizes = [4, 8] if args.smoke else args.sizes
    rows: list[dict[str, Any]] = []
    for memory in sizes:
        cases, peak = host_allocation_peak(lambda: _case(memory, args.repeats))
        for row in cases:
            row["host_allocation_peak_bytes"] = peak
        rows.extend(cases)
    comparisons: dict[str, Any] = {}
    for family, baseline_route in (("projection", "selector"), ("contract", "kronecker")):
        baseline = [
            row["compiler"]["temporary_bytes"]
            for row in rows
            if row["family"] == family and row["route"] == baseline_route
        ]
        candidate = [
            row["compiler"]["temporary_bytes"]
            for row in rows
            if row["family"] == family and row["route"] == "production"
        ]
        comparisons[family] = {
            "memory": evaluate_memory_gate(
                sizes, baseline, candidate, MemoryGatePolicy()
            ).to_dict(),
            "baseline_exponent": fitted_scaling_exponent(sizes, baseline),
            "candidate_exponent": fitted_scaling_exponent(sizes, candidate),
            "adoption_rule": "unconditional-explicit-selector-identity-removal",
            "declared_state_workspace_exponent": 2.0,
        }
    fields = (
        "environment",
        "configuration",
        "cases",
        "comparisons",
        "process_peak_rss_bytes",
    )
    record = {
        "identity": capture_benchmark_identity(
            Path.cwd(), Path(__file__), fields
        ).to_dict(),
        "environment": asdict(capture_environment()),
        "configuration": {
            "capacities": sizes,
            "repeats": args.repeats,
            "timing_acceptance": "deferred-quiet-serialized-wave",
        },
        "cases": rows,
        "comparisons": comparisons,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }
    write_json_atomic(args.output, record)
    print(comparisons)


if __name__ == "__main__":
    main()

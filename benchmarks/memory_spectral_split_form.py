"""Phase-separated Fourier split-form preparation and bounded pair workspace.

Small defaults are a smoke campaign; use --sizes 32 64 128 256 --chunks 512
for retained-storage scaling. Timing under contention is evidence, not acceptance.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

import phydrax as phx
from phydrax.discretization.spectral import PreparedSpectralConservationDynamics

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


def _residual(dynamics: PreparedSpectralConservationDynamics, state: Array) -> Array:
    return dynamics(jnp.asarray(0.0, dtype=jnp.float64), state)


class _Compiled(Protocol):
    compiled: jax.stages.Compiled

    def __call__(
        self, dynamics: PreparedSpectralConservationDynamics, state: Array, /
    ) -> Array: ...


class _Lowered(Protocol):
    def compile(self) -> _Compiled: ...


@runtime_checkable
class _Entry(Protocol):
    def lower(
        self, dynamics: PreparedSpectralConservationDynamics, state: Array, /
    ) -> _Lowered: ...


def _require_entry(
    entry: Callable[[PreparedSpectralConservationDynamics, Array], Array], /
) -> _Entry:
    if not isinstance(entry, _Entry):
        raise TypeError("Spectral benchmark requires a callable lowering entry.")
    return entry


_ENTRY = _require_entry(eqx.filter_jit(_residual))


def _case(count: int, chunk: int, repeats: int) -> dict[str, Any]:
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="state",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 2.0),))
    system = phx.equations.EulerSystem(1)
    entropy = phx.equations.ideal_gas_euler_entropy_pair(system)
    method = phx.discretization.SpectralConservationMethodPlan(
        split_form=phx.discretization.SpectralSplitFormPlan(
            phx.discretization.EntropyConservativeEulerFluxPlan(),
            pair_chunk_size=chunk,
        ),
        entropy_diagnostics=True,
    )
    started = time.perf_counter()
    dynamics, host_peak = host_allocation_peak(
        lambda: PreparedSpectralConservationDynamics(
            system, space, method, entropy_pair=entropy
        )
    )
    preparation_seconds = time.perf_counter() - started
    x = space.axes[0].nodes
    physical = system.primitive_to_conserved(
        jnp.stack(
            (
                1.0 + 0.05 * jnp.sin(jnp.pi * x),
                0.2 * jnp.cos(jnp.pi * x),
                1.0 + 0.03 * jnp.sin(2.0 * jnp.pi * x),
            ),
            axis=-1,
        )
    )
    state = space.project(physical)
    executable, timing = measure_lower_and_compile(
        lambda: _ENTRY.lower(dynamics, state), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(
        lambda: executable(dynamics, state), warmup=1, repeats=repeats
    )
    evidence = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="jax-fourier-split-form",
        unavailable_reason="Compiler did not supply all estimates.",
    )
    _, diagnostics = dynamics.residual_with_diagnostics(
        jnp.asarray(0.0, dtype=jnp.float64), state
    )
    split = dynamics.method.split_form
    if split is None or diagnostics.entropy is None:
        raise RuntimeError("Split form did not expose its report and entropy evidence.")
    report = split.report
    return {
        "count": count,
        "chunk": chunk,
        "retained_split_bytes": logical_array_bytes(split),
        "host_preparation_peak_bytes": host_peak,
        "preparation_seconds": preparation_seconds,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "output_bytes": logical_array_bytes(result),
        "pair_count": report.pair_count,
        "pair_workspace_bytes": report.pair_workspace_bytes,
        "skew_sbp_defect": report.skew_sbp_defect,
        "constant_state_defect": report.constant_state_defect,
        "report_id": report.report_id,
        "conservation_defect": float(jnp.max(jnp.abs(diagnostics.conservation_defect))),
        "entropy_defect": float(diagnostics.entropy.convective_entropy_defect),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--chunks", type=int, nargs="+", default=[128])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    args = parser.parse_args()
    if len(args.sizes) < 2 or sorted(set(args.sizes)) != args.sizes:
        parser.error("sizes must contain at least two increasing distinct capacities")
    cases = [
        _case(count, chunk, args.repeats) for chunk in args.chunks for count in args.sizes
    ]
    scaling = []
    for chunk in args.chunks:
        rows = [row for row in cases if row["chunk"] == chunk]
        scaling.append(
            {
                "chunk": chunk,
                "retained_exponent": fitted_scaling_exponent(
                    args.sizes, [row["retained_split_bytes"] for row in rows]
                ),
            }
        )
    gates = []
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        for chunk in args.chunks:
            old = [row for row in baseline["cases"] if row["chunk"] == chunk]
            new = [row for row in cases if row["chunk"] == chunk]
            if [row["count"] for row in old] != args.sizes:
                parser.error("baseline capacities do not match")
            gates.append(
                {
                    "chunk": chunk,
                    "scope": "compiler-only diagnostic; retained fix is unconditional",
                    **evaluate_memory_gate(
                        args.sizes,
                        [row["compiler"]["temporary_bytes"] for row in old],
                        [row["compiler"]["temporary_bytes"] for row in new],
                        MemoryGatePolicy(),
                    ).to_dict(),
                }
            )
    fields = ("environment", "configuration", "cases", "scaling", "compiler_gates")
    root = Path(__file__).resolve().parents[1]
    record = {
        "identity": capture_benchmark_identity(root, Path(__file__), fields).to_dict(),
        "environment": capture_environment().to_dict(),
        "configuration": {
            "sizes": args.sizes,
            "chunks": args.chunks,
            "repeats": args.repeats,
        },
        "cases": cases,
        "scaling": scaling,
        "compiler_gates": gates,
    }
    write_json_atomic(args.output, record)
    print(
        json.dumps(
            {"output": str(args.output), "scaling": scaling, "compiler_gates": gates}
        )
    )


if __name__ == "__main__":
    main()

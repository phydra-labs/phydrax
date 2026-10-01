# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded point-capacity scaling with separately measured preparation/execution."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    benchmark_driver_fingerprint,
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax._fingerprint import canonical_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    "neighbor",
    "stencil",
    "assembly",
    "hierarchy",
    "lowering",
    "compilation",
    "first",
    "warm",
    "numeric-refresh",
    "epoch-transition",
    "adjoint",
)


@dataclass(frozen=True, slots=True)
class MeshfreeConfig:
    """Explicit workload and refusal limits; sizes are total point capacities."""

    sizes: tuple[int, ...] = (128, 256)
    seeds: tuple[int, ...] = (0,)
    dimension: int = 2
    repeats: int = 3
    neighbors: int = 24
    degree: int = 2
    chunk_rows: int = 32
    working_set_bytes: int = 256 * 1024 * 1024
    resource_bytes: int = 1024 * 1024 * 1024
    max_points: int = 4096
    steps: int = 80
    precision: str = "float64"

    def __post_init__(self) -> None:
        if self.dimension not in (2, 3) or self.precision not in ("float32", "float64"):
            raise ValueError(
                "dimension must be 2 or 3; precision must be float32 or float64."
            )
        integers = (
            *self.sizes,
            self.repeats,
            self.neighbors,
            self.chunk_rows,
            self.working_set_bytes,
            self.resource_bytes,
            self.max_points,
            self.steps,
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError(
                "Sizes, repeats, capacities, and budgets must be positive integers."
            )
        if (
            not self.sizes
            or not self.seeds
            or len(set(self.sizes)) != len(self.sizes)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError(
                "Provide nonempty distinct sizes and nonempty distinct seeds."
            )
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("Seeds must be nonnegative integers.")
        if type(self.degree) is not int or self.degree < 2:
            raise ValueError("degree must be an integer at least two.")
        basis = comb(self.dimension + self.degree, self.degree)
        if self.neighbors < basis or min(self.sizes) < self.neighbors:
            raise ValueError(
                "Every requested capacity must support the requested polynomial neighborhood."
            )
        for capacity in self.sizes:
            self.check_capacity(capacity)

    def check_capacity(
        self, capacity: int, /, *, degree: int | None = None, neighbors: int | None = None
    ) -> int:
        """Refuse requested workloads using their actual local-fit policy."""
        selected_degree = self.degree if degree is None else degree
        selected_neighbors = self.neighbors if neighbors is None else neighbors
        if (
            type(capacity) is not int
            or capacity < 1
            or type(selected_degree) is not int
            or selected_degree < 2
        ):
            raise ValueError(
                "Workload capacity and polynomial degree must be positive supported integers."
            )
        basis = comb(self.dimension + selected_degree, selected_degree)
        if (
            type(selected_neighbors) is not int
            or selected_neighbors < basis
            or capacity < selected_neighbors
        ):
            raise ValueError(
                "Workload capacity and neighborhood must cover the selected polynomial basis."
            )
        # This is a planning reservation, not a certified or measured peak bound.
        # Include relations, local systems, candidate search and simultaneous factors.
        reservation = 8 * (
            capacity * selected_neighbors * (self.dimension + basis + 12)
            + self.chunk_rows * (selected_neighbors + basis) ** 2 * 16
            + capacity * self.chunk_rows * (self.dimension + 8)
        )
        if capacity > self.max_points or reservation > self.working_set_bytes:
            raise ValueError(
                f"Requested {capacity} points reserves {reservation} bytes; workload exceeds budget."
            )
        if reservation > self.resource_bytes:
            raise ValueError("Requested workload exceeds total resource budget.")
        return reservation


def add_config_arguments(parser: argparse.ArgumentParser, /) -> None:
    """Share unambiguous qualification and benchmark command-line controls."""
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[128, 256],
        help="Total point capacities, never points per axis",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--neighbors", type=int, default=24, help="Generic bulk/benchmark stencil width"
    )
    parser.add_argument(
        "--degree", type=int, default=2, help="Generic bulk/benchmark polynomial degree"
    )
    parser.add_argument("--chunk-rows", type=int, default=32)
    parser.add_argument("--working-set-mib", type=int, default=256)
    parser.add_argument("--resource-mib", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Actual compatible benchmark record for explicit before/after fields",
    )


def config_from_arguments(arguments: argparse.Namespace, /) -> MeshfreeConfig:
    return MeshfreeConfig(
        sizes=tuple(arguments.sizes),
        seeds=tuple(arguments.seeds),
        dimension=arguments.dimension,
        repeats=arguments.repeats,
        neighbors=arguments.neighbors,
        degree=arguments.degree,
        chunk_rows=arguments.chunk_rows,
        working_set_bytes=arguments.working_set_mib * 1024**2,
        resource_bytes=arguments.resource_mib * 1024**2,
        max_points=arguments.max_points,
        steps=arguments.steps,
        precision=arguments.precision,
    )


def cloud_points(capacity: int, dimension: int, seed: int, /) -> np.ndarray:
    """A deterministic stratified cloud with exactly the controlling capacity."""
    from scipy.stats import qmc

    return qmc.LatinHypercube(d=dimension, seed=seed).random(capacity)


def unavailable_phases() -> dict[str, Any]:
    return {
        phase: {"status": "unavailable", "reason": "Not a phase of this native operation"}
        for phase in PHASES
    }


def measured_phase(seconds: float, /) -> dict[str, object]:
    return {"status": "measured", "seconds": seconds}


def execution_evidence(
    action: Callable[[Array], Array], values: Array, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Own all lowering/compiler/warm timing, never estimate compilation counts."""

    def apply(value: Array) -> Array:
        return action(value)

    compiled, timing = measure_lower_and_compile(
        lambda: jax.jit(apply).lower(values), lambda lowered: lowered.compile()
    )
    compiler = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-meshfree-consumer",
        unavailable_reason="Compiler did not expose every estimate",
    )
    estimate = compiler.estimated_device_memory_bytes
    if estimate is not None and estimate > config.resource_bytes:
        raise ValueError(
            f"Compiler resident estimate {estimate} exceeds resource budget."
        )
    first, first_seconds = measure_synchronized(lambda: compiled(values))
    result, warm = measure_repeated(
        lambda: compiled(values), warmup=1, repeats=config.repeats
    )
    adjoint = jax.jit(jax.grad(lambda value: jnp.sum(jnp.square(apply(value)))))
    adjoint_compiled, adjoint_timing = measure_lower_and_compile(
        lambda: adjoint.lower(values), lambda lowered: lowered.compile()
    )
    adjoint_compiler = compiler_evidence(
        adjoint_compiled.cost_analysis(),
        adjoint_compiled.memory_analysis(),
        source="jax-compiled-meshfree-adjoint",
        unavailable_reason="Compiler did not expose every estimate",
    )
    if (
        adjoint_compiler.estimated_device_memory_bytes is not None
        and adjoint_compiler.estimated_device_memory_bytes > config.resource_bytes
    ):
        raise ValueError("Adjoint compiler resident estimate exceeds resource budget.")
    _, adjoint_first = measure_synchronized(lambda: adjoint_compiled(values))
    _, adjoint_warm = measure_repeated(
        lambda: adjoint_compiled(values), warmup=1, repeats=config.repeats
    )
    return {
        "phases": {
            "lowering": measured_phase(timing.lowering_seconds),
            "compilation": measured_phase(timing.compilation_seconds),
            "first": measured_phase(first_seconds),
            "warm": {"status": "measured", **warm.to_seconds_dict()},
            "adjoint": {
                "status": "measured",
                "lowering_seconds": adjoint_timing.lowering_seconds,
                "compilation_seconds": adjoint_timing.compilation_seconds,
                "first_seconds": adjoint_first,
                "warm": adjoint_warm.to_seconds_dict(),
                "compiler": asdict(adjoint_compiler),
            },
        },
        "compiler": asdict(compiler),
        "output_bytes": logical_array_bytes(result),
        "first_output_bytes": logical_array_bytes(first),
        "result": result,
    }


def measure_capacity(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    from phydrax.discretization.meshfree import (
        LocalStencilPolicy,
        MeshfreeFunctional,
        MeshfreeNeighborhoodPlan,
        prepare_local_stencils,
    )

    reservation = config.check_capacity(capacity)
    points = cloud_points(capacity, config.dimension, seed)
    neighborhood, neighbor_seconds = measure_synchronized(
        lambda: MeshfreeNeighborhoodPlan(
            points, config.neighbors, target_chunk_size=config.chunk_rows
        ).prepare()
    )
    indices = tuple(
        tuple(2 if axis == d else 0 for axis in range(config.dimension))
        for d in range(config.dimension)
    )
    functional = MeshfreeFunctional(indices, np.ones(config.dimension), name="laplacian")
    prepared, stencil_seconds = measure_synchronized(
        lambda: prepare_local_stencils(
            neighborhood,
            points,
            points,
            (functional,),
            LocalStencilPolicy(
                polynomial_degree=config.degree, chunk_rows=config.chunk_rows
            ),
        )
    )
    relation = prepared.neighborhood.relation

    def action(values: Array) -> Array:
        return jnp.sum(
            jnp.where(
                relation.valid, prepared.weights[0] * values[relation.source_indices], 0
            ),
            axis=1,
        )

    values = jnp.asarray(np.sum(points**2, axis=1))
    execution = execution_evidence(action, values, config)
    defect = float(
        np.max(np.abs(np.asarray(execution.pop("result")) - 2 * config.dimension))
    )
    phases = unavailable_phases()
    phases.update(execution.pop("phases"))
    phases.update(
        neighbor=measured_phase(neighbor_seconds), stencil=measured_phase(stencil_seconds)
    )
    retained = logical_array_bytes((points, prepared))
    if retained > config.working_set_bytes:
        raise ValueError("Actual retained arrays exceed working-set budget.")
    tolerance = 5e-3 if config.precision == "float32" else 1e-7
    if defect > tolerance:
        raise AssertionError(f"Polynomial Laplacian defect {defect} exceeds {tolerance}.")
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "domain": "unit-cube-stratified",
        "phases": phases,
        **execution,
        "retained_bytes": retained,
        "reserved_working_set_bytes": reservation,
        "polynomial_laplacian_defect": defect,
        "prepared_id": prepared.prepared_id,
        "oracle": "independent-host-NumPy: Laplacian(sum(x_i^2))=2d",
    }


def make_record(
    config: MeshfreeConfig,
    rows: Sequence[dict[str, Any]],
    driver: Path,
    /,
    *,
    consumers: tuple[Path, ...] = (),
) -> dict[str, Any]:
    configuration = {
        **asdict(config),
        "sizes": list(config.sizes),
        "seeds": list(config.seeds),
    }
    record: dict[str, Any] = {
        "kind": "meshfree-phase-benchmark",
        "config": configuration,
        "rows": list(rows),
        "environment": capture_environment().to_dict(),
        "precision": config.precision,
        "release_status": "candidate",
        "resource_measurement": {
            "retained_scope": "unique logical payload visible to runtime object walker",
            "compiler_scope": "official executable estimates, not process resident memory",
            "peak_resident_bytes": None,
            "peak_status": "unavailable",
            "reason": "No native peak-memory sampler is exposed by the benchmark runtime",
        },
        "driver_dependencies": {
            path.relative_to(PROJECT_ROOT).as_posix(): benchmark_driver_fingerprint(
                PROJECT_ROOT, path
            )
            for path in (Path(__file__), *consumers)
        },
    }
    identity = capture_benchmark_identity(PROJECT_ROOT, driver, tuple(record))
    record["identity"] = identity.to_dict()
    record["workload_id"] = canonical_fingerprint(
        {"config": configuration, "driver": driver.name}
    )
    return record


def apply_baseline(record: dict[str, Any], baseline_path: Path | None, /) -> None:
    """Only retain before/after evidence when supplied by an actual prior run."""
    if baseline_path is None:
        return
    with baseline_path.open(encoding="utf-8") as stream:
        baseline = json.load(stream)
    for field in ("kind", "config", "workload_id", "environment"):
        if baseline.get(field) != record[field]:
            raise ValueError(f"Baseline has incompatible {field}.")
    if not isinstance(baseline.get("rows"), list) or not baseline.get("identity"):
        raise ValueError("Baseline lacks recorded rows and source identity.")
    record["performance_before_after"] = {
        "baseline_path": str(baseline_path),
        "before_identity": baseline["identity"],
        "before": baseline["rows"],
        "after": record["rows"],
    }


def run(config: MeshfreeConfig = MeshfreeConfig(), /) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", config.precision == "float64")
    return make_record(
        config,
        [
            measure_capacity(size, seed, config)
            for size in config.sizes
            for seed in config.seeds
        ],
        Path(__file__),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    args = parser.parse_args()
    record = run(config_from_arguments(args))
    apply_baseline(record, args.baseline)
    if args.output is not None:
        write_json_atomic(args.output, record)
    else:
        print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

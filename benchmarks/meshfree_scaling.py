# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded point-capacity scaling with separately measured preparation/execution.

This module also owns the shared meshfree evidence utilities (phase vocabulary,
phase recorder, declared capacity admission, measured fill distance, and the
record writer) used by every meshfree benchmark and qualification driver.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections.abc import Callable, Sequence
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field, replace
from math import comb
from pathlib import Path
from threading import get_ident
from types import TracebackType
from typing import Any, get_args, Literal, TypeAlias, TypeVar

import equinox as eqx
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
    measure_synchronized,
)
from phydrax._fingerprint import canonical_fingerprint
from phydrax.execution import compiled_memory_estimate, PhaseMemorySampler
from phydrax.typing import parse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_T = TypeVar("_T")

MeshfreePhase: TypeAlias = Literal[
    "search",
    "geometry",
    "local-fit",
    "rank-certificate",
    "conic",
    "ordering-fill",
    "hierarchy",
    "transfer",
    "assembly",
    "lowering",
    "compilation",
    "first",
    "warm",
    "solve",
    "numeric-refresh",
    "jvp",
    "vjp",
    "epoch-commit",
    "communication-migration",
    "restart",
    "output",
    "end-to-end",
]
PHASES: tuple[MeshfreePhase, ...] = get_args(MeshfreePhase)
_BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


class _CompileMonitor:
    """Process-wide XLA backend compilation count and seconds from jax.monitoring."""

    __slots__ = ("count", "seconds")

    def __init__(self) -> None:
        self.count = 0
        self.seconds = 0.0

    def __call__(self, event: str, duration_secs: float, **_: str | int) -> None:
        if event == _BACKEND_COMPILE_EVENT:
            self.count += 1
            self.seconds += duration_secs


_COMPILES = _CompileMonitor()
jax.monitoring.register_event_duration_secs_listener(_COMPILES)


class DeclaredCapacityRefusal(ValueError):
    """Declared or observed workload resources exceed the supplied budget."""


@dataclass(slots=True)
class BenchmarkExecutionScope:
    """Own invocation-local evidence across a benchmark failure boundary.

    Nested invocations restore the enclosing scope. Copied contexts cannot
    attach another thread's recorders or reservations.
    """

    recorders: list[PhaseRecorder] = field(default_factory=list)
    reservations: list[dict[str, Any]] = field(default_factory=list)
    executed: bool = False
    _thread: int = field(default_factory=get_ident)
    _identity: object = field(default_factory=object)
    _used: bool = False
    _token: Token[BenchmarkExecutionScope | None] | None = None

    def __enter__(self) -> BenchmarkExecutionScope:
        if self._used or self._thread != get_ident():
            raise RuntimeError(
                "Benchmark execution scopes cannot be reused or transferred."
            )
        self._used = True
        self._token = _EXECUTION_SCOPE.set(self)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        token = self._token
        if token is None:
            raise RuntimeError("Benchmark execution scope is not active.")
        _EXECUTION_SCOPE.reset(token)
        self._token = None

    def record(self) -> dict[str, Any]:
        partial: dict[str, Any] = {
            "execution_stage": "post-admission" if self.executed else "pre-admission",
            "reservations": list(self.reservations),
        }
        admitted = [
            item["reserved_working_set_bytes"]
            for item in self.reservations
            if item["status"] == "admitted"
        ]
        if admitted:
            partial["reserved_working_set_bytes"] = max(admitted)
            partial["reservation_scope"] = (
                "largest individually admitted reservation; not an aggregate bound"
            )
        if self.recorders:
            partial["phases"] = _retained_phases(self.recorders)
        return partial


_EXECUTION_SCOPE: ContextVar[BenchmarkExecutionScope | None] = ContextVar(
    "meshfree_benchmark_execution_scope", default=None
)


def _execution_scope() -> BenchmarkExecutionScope | None:
    scope = _EXECUTION_SCOPE.get()
    return scope if scope is not None and scope._thread == get_ident() else None


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
    candidates: int | None = None

    def __post_init__(self) -> None:
        if self.dimension not in (1, 2, 3) or self.precision not in (
            "float32",
            "float64",
        ):
            raise ValueError(
                "dimension must be 1, 2, or 3; precision must be float32 or float64."
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
        if self.neighbors < basis:
            raise ValueError(
                "The neighborhood capacity must cover the requested polynomial basis."
            )

    def check_point_limit(self, capacity: int, /) -> None:
        """Admit point counts independently of a workload's local-fit policy."""
        if type(capacity) is not int:
            raise TypeError("Point capacity must be an integer.")
        if capacity < 1:
            raise ValueError("Point capacity must be positive.")
        if capacity > self.max_points:
            raise DeclaredCapacityRefusal(
                f"Requested {capacity} points exceeds the declared max_points={self.max_points}."
            )

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
        self.check_point_limit(capacity)
        return declare_reservation(reservation, self, scope="local-fit planning")


def declare_reservation(
    reserved_bytes: int, config: MeshfreeConfig, /, *, scope: str
) -> int:
    """Admit a declared reservation against budgets supplied before execution."""
    if type(reserved_bytes) is not int or reserved_bytes < 0:
        raise ValueError("A declared reservation must be a nonnegative integer.")
    budget = min(config.working_set_bytes, config.resource_bytes)
    execution = _execution_scope()
    if execution is not None:
        execution.reservations.append(
            {
                "scope": scope,
                "reserved_working_set_bytes": reserved_bytes,
                "budget_bytes": budget,
                "status": "refused" if reserved_bytes > budget else "admitted",
            }
        )
    if reserved_bytes > budget:
        raise DeclaredCapacityRefusal(
            f"{scope} reserves {reserved_bytes} bytes; declared budget is {budget} bytes."
        )
    return reserved_bytes


def configure_precision(config: MeshfreeConfig, /, *, supported: tuple[str, ...]) -> None:
    """Refuse an undeclared data precision before execution; keep 64-bit addressing.

    Meshfree Morton codes, sort keys and stable IDs are 64-bit for every data
    precision, so x64 stays enabled and ``precision`` is the declared
    coordinate/field dtype that each workload must honor.
    """
    if config.precision not in supported:
        raise ValueError(
            f"Precision {config.precision} is not declared for this workload "
            f"(supported: {', '.join(supported)})."
        )
    jax.config.update("jax_enable_x64", True)


class PhaseRecorder:
    """Accumulate phase-separated wall, process-CPU, compile and sampled-memory evidence.

    Every occurrence synchronizes its result before the clock stops. CPU time is
    process-wide (all threads), which on a loaded host is reported beside wall
    time rather than used to correct it. Compile counts are XLA backend compiles
    observed through ``jax.monitoring`` during the occurrence. Memory is the
    native ``PhaseMemorySampler`` sampled resident peak, never an upper bound.
    """

    def __init__(self, *, memory_interval_seconds: float = 0.05) -> None:
        self._interval = memory_interval_seconds
        self._occurrences: dict[MeshfreePhase, list[dict[str, Any]]] = {}
        self._memory: dict[MeshfreePhase, dict[str, Any]] = {}
        self._extras: dict[MeshfreePhase, dict[str, Any]] = {}
        self._unavailable: dict[MeshfreePhase, str] = {}
        self._failures: dict[MeshfreePhase, dict[str, Any]] = {}
        execution = _execution_scope()
        self._owner_scope_token = None if execution is None else execution._identity
        self._owner_thread = get_ident()
        if execution is not None:
            execution.recorders.append(self)

    def _admit_scope(self) -> BenchmarkExecutionScope | None:
        execution = _execution_scope()
        scope_token = None if execution is None else execution._identity
        if (
            self._owner_scope_token is not scope_token
            or self._owner_thread != get_ident()
        ):
            raise RuntimeError(
                "Phase recorders cannot cross benchmark invocation or thread boundaries."
            )
        return execution

    def run(
        self,
        phase: MeshfreePhase,
        operation: Callable[[], _T],
        /,
        *,
        scope: str | None = None,
    ) -> _T:
        """Measure one synchronized occurrence of ``phase`` and return its value."""
        name = parse(phase, MeshfreePhase, "phase")
        execution = self._admit_scope()
        devices = tuple(
            device for device in jax.local_devices() if device.platform != "cpu"
        )
        compiles, compile_seconds = _COMPILES.count, _COMPILES.seconds
        cpu = time.process_time()
        with PhaseMemorySampler(
            name, interval_seconds=self._interval, devices=devices
        ) as sampler:
            if execution is not None:
                execution.executed = True
            value, wall = measure_synchronized(operation)
        evidence = sampler.evidence
        increase = evidence.sampled_peak_increase_bytes
        occurrence = {
            "scope": scope,
            "wall_seconds": wall,
            "cpu_seconds": time.process_time() - cpu,
            "compile_count": _COMPILES.count - compiles,
            "compile_seconds": _COMPILES.seconds - compile_seconds,
            "sampled_peak_increase_bytes": increase,
        }
        self._occurrences.setdefault(name, []).append(occurrence)
        retained = self._memory.get(name)
        if retained is None or (
            increase is not None
            and (
                retained["sampled_peak_increase_bytes"] is None
                or increase > retained["sampled_peak_increase_bytes"]
            )
        ):
            self._memory[name] = {
                "sampled_peak_increase_bytes": increase,
                "evidence": evidence.to_payload(),
            }
        return value

    def annotate(self, phase: MeshfreePhase, /, **values: Any) -> None:
        """Attach measured phase-specific evidence (e.g. timing distributions)."""
        name = parse(phase, MeshfreePhase, "phase")
        self._admit_scope()
        self._extras.setdefault(name, {}).update(values)

    def unavailable(self, phase: MeshfreePhase, reason: str, /) -> None:
        """Declare why a vocabulary phase is not separately measurable here."""
        name = parse(phase, MeshfreePhase, "phase")
        self._admit_scope()
        if not reason:
            raise ValueError("An unavailable phase needs an explicit reason.")
        self._unavailable[name] = reason

    def run_repeated(
        self,
        phase: MeshfreePhase,
        operation: Callable[[], _T],
        /,
        *,
        repeats: int,
        scope: str | None = None,
    ) -> _T:
        """Measure ``repeats`` synchronized occurrences; retain every sample."""
        if repeats < 1:
            raise ValueError("repeats must be positive.")
        value = self.run(phase, operation, scope=scope)
        for _ in range(repeats - 1):
            value = self.run(phase, operation, scope=scope)
        return value

    def compiled_action(
        self,
        action: Callable[[Array], Array],
        values: Array,
        /,
        *,
        budget_bytes: int,
        repeats: int,
        derivatives: bool = True,
        scope: str = "primal",
    ) -> tuple[Array, dict[str, Any]]:
        """Lower, compile, first/warm execute, then JVP/VJP of one array action."""
        evidence: dict[str, Any] = {}

        def compile_route(
            route: Callable[..., Any], arguments: tuple[Any, ...], label: str
        ) -> tuple[Callable[..., Any], Callable[..., Any]]:
            dynamic, static = eqx.partition(route, eqx.is_array)

            def apply(dynamic_route: Any, *values: Any) -> Any:
                return eqx.combine(dynamic_route, static)(*values)

            jitted = jax.jit(apply)
            lowered = self.run(
                "lowering",
                lambda: jitted.lower(dynamic, *arguments),
                scope=label,
            )
            compiled = self.run("compilation", lowered.compile, scope=label)
            compiler = compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source=f"jax-compiled-meshfree-{label}",
                unavailable_reason="Compiler did not expose every estimate",
            )
            estimate = compiled_memory_estimate(compiled)
            evidence[label] = {
                **asdict(compiler),
                "peak_bytes": None if estimate is None else estimate.peak_bytes,
            }
            self.annotate("compilation", compiler=dict(evidence))
            device_bytes = compiler.estimated_device_memory_bytes
            if device_bytes is not None and device_bytes > budget_bytes:
                raise DeclaredCapacityRefusal(
                    f"Compiler {label} resident estimate {device_bytes} exceeds "
                    f"the declared budget {budget_bytes}."
                )

            def execute(*values: Any) -> Any:
                return compiled(dynamic, *values)

            return execute, eqx.Partial(jitted, dynamic)

        compiled, prepared_action = compile_route(action, (values,), scope)
        self.run("first", lambda: compiled(values), scope=scope)
        result = self.run_repeated(
            "warm", lambda: compiled(values), repeats=repeats, scope=scope
        )
        output_bytes = logical_array_bytes(result)
        self.annotate("warm", output_bytes=output_bytes)
        derivative_status: dict[str, str] = {}
        if derivatives:
            tangent = jnp.ones_like(values)

            def jvp_action(
                bound_action: Callable[[Array], Array], value: Array, direction: Array
            ) -> tuple[Array, Array]:
                return jax.jvp(bound_action, (value,), (direction,))

            def vjp_action(
                bound_action: Callable[[Array], Array], value: Array, weight: Array
            ) -> Array:
                return jax.vjp(bound_action, value)[1](weight)

            cotangent = jnp.ones_like(result)
            routes: tuple[
                tuple[MeshfreePhase, Callable[..., Any], tuple[Array, Array]], ...
            ] = (
                (
                    "jvp",
                    eqx.Partial(jvp_action, prepared_action),
                    (values, tangent),
                ),
                (
                    "vjp",
                    eqx.Partial(vjp_action, prepared_action),
                    (values, cotangent),
                ),
            )
            for name, route, arguments in routes:
                try:
                    executable, _ = compile_route(route, arguments, f"{scope}-{name}")
                    self.run_repeated(
                        name,
                        lambda: executable(*arguments),
                        repeats=repeats,
                        scope=scope,
                    )
                except DeclaredCapacityRefusal:
                    raise
                except (
                    Exception
                ) as error:  # A refused derivative is evidence; primal stays.
                    self._failures[name] = failure_record(error)
                    derivative_status[name] = "failed"
                else:
                    derivative_status[name] = "measured"
        return result, {
            "compiler": evidence,
            "output_bytes": output_bytes,
            "derivatives": derivative_status,
        }

    def record(self) -> dict[str, Any]:
        """Return every vocabulary phase as measured, unavailable, or not separated."""
        phases: dict[str, Any] = {}
        for phase in PHASES:
            occurrences = self._occurrences.get(phase)
            if occurrences is None:
                phases[phase] = {
                    "status": "unavailable",
                    "reason": self._unavailable.get(
                        phase, "Not a separately executed phase of this workload"
                    ),
                    **self._extras.get(phase, {}),
                }
                if phase in self._failures:
                    phases[phase].update(status="failed", failure=self._failures[phase])
                continue
            phases[phase] = {
                "status": "measured",
                "occurrences": occurrences,
                "wall_seconds": sum(item["wall_seconds"] for item in occurrences),
                "cpu_seconds": sum(item["cpu_seconds"] for item in occurrences),
                "compile_count": sum(item["compile_count"] for item in occurrences),
                "compile_seconds": sum(item["compile_seconds"] for item in occurrences),
                "memory": self._memory[phase],
                **self._extras.get(phase, {}),
            }
            if phase in self._failures:
                phases[phase].update(status="failed", failure=self._failures[phase])
        return phases


def _retained_phases(recorders: Sequence[PhaseRecorder], /) -> dict[str, Any]:
    """Merge native phase evidence; retain annotations from every recorder."""
    if len(recorders) == 1:
        return recorders[0].record()
    records = [recorder.record() for recorder in recorders]
    phases: dict[str, Any] = {}
    for phase in PHASES:
        sources = [record[phase] for record in records]
        combined = dict(sources[-1])
        annotations = [
            recorder._extras[phase] for recorder in recorders if phase in recorder._extras
        ]
        for annotation in annotations:
            compiler = {
                **combined.get("compiler", {}),
                **annotation.get("compiler", {}),
            }
            combined.update(annotation)
            if compiler:
                combined["compiler"] = compiler
        if annotations:
            combined["annotations"] = annotations
        occurrences = [
            item for source in sources for item in source.get("occurrences", ())
        ]
        if occurrences:
            combined.pop("reason", None)
            combined.update(status="measured", occurrences=occurrences)
            for key in (
                "wall_seconds",
                "cpu_seconds",
                "compile_count",
                "compile_seconds",
            ):
                combined[key] = sum(item[key] for item in occurrences)
            memories = [source["memory"] for source in sources if "memory" in source]
            combined["memory"] = max(
                memories,
                key=lambda memory: (
                    -1
                    if memory["sampled_peak_increase_bytes"] is None
                    else memory["sampled_peak_increase_bytes"]
                ),
            )
        failures = [source["failure"] for source in sources if "failure" in source]
        if failures:
            combined.update(status="failed", failure=failures[-1])
            if len(failures) > 1:
                combined["failures"] = failures
        phases[phase] = combined
    return phases


def failure_record(error: BaseException, /) -> dict[str, Any]:
    """Retain an executed failure verbatim; no replacement value or retry."""
    return {
        "exception": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exception(error),
    }


def fill_distance(
    points: np.ndarray, probes: np.ndarray, /, *, provenance: str
) -> dict[str, Any]:
    """Measured fill distance h and separation q of a cloud, with provenance.

    ``h = max_probe min_point |probe - point|`` over supplied probes of the
    declared domain (a lower estimate of the true supremum), and
    ``q = 0.5 min_{i != j} |x_i - x_j|``.
    """
    from scipy.spatial import cKDTree

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim == 1:
        cloud = cloud[:, None]
    probe = np.asarray(probes, dtype=np.float64)
    if probe.ndim == 1:
        probe = probe[:, None]
    tree = cKDTree(cloud)
    nearest, _ = tree.query(probe)
    pairs, _ = tree.query(cloud, k=2)
    return {
        "fill_distance": float(np.max(nearest)),
        "separation_distance": float(0.5 * np.min(pairs[:, 1])),
        "probe_count": probe.shape[0],
        "point_count": cloud.shape[0],
        "provenance": provenance,
    }


def unit_cube_probes(dimension: int, count: int, seed: int, /) -> np.ndarray:
    """Scrambled Sobol probes of the unit cube for measured fill distance."""
    from scipy.stats import qmc

    exponent = max(int(np.ceil(np.log2(count))), 1)
    return qmc.Sobol(d=dimension, scramble=True, seed=seed + 7919).random_base2(exponent)


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
    parser.add_argument("--dimension", type=int, choices=(1, 2, 3), default=2)
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
    parser.add_argument(
        "--candidates",
        type=int,
        help="Per-target neighbor candidate capacity; default is the native owner's",
    )
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
        candidates=arguments.candidates,
    )


def cloud_points(capacity: int, dimension: int, seed: int, /) -> np.ndarray:
    """A deterministic stratified cloud with exactly the controlling capacity."""
    from scipy.stats import qmc

    return qmc.LatinHypercube(d=dimension, seed=seed).random(capacity)


def execution_evidence(
    action: Callable[[Array], Array],
    values: Array,
    config: MeshfreeConfig,
    recorder: PhaseRecorder,
    /,
) -> dict[str, Any]:
    """Record lowering/compile/first/warm/JVP/VJP phases of one compiled action."""
    result, evidence = recorder.compiled_action(
        action,
        values,
        budget_bytes=config.resource_bytes,
        repeats=config.repeats,
    )
    return {**evidence, "result": result}


def refusal_row(
    capacity: int, seed: int, config: MeshfreeConfig, error: DeclaredCapacityRefusal, /
) -> dict[str, Any]:
    """A declared refusal is resource evidence, never an accepted measurement."""
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "neighbor_capacity": config.neighbors,
        "chunk_rows": config.chunk_rows,
        "status": "declared-refusal",
        "reason": str(error),
        "working_set_budget": config.working_set_bytes,
        "resource_budget": config.resource_bytes,
        "max_points": config.max_points,
    }


def measure_capacity(
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    /,
) -> dict[str, Any]:
    from phydrax.discretization.meshfree import (
        LocalStencilPolicy,
        MeshfreeFunctional,
        MeshfreeNeighborhoodPlan,
        MeshfreePrecisionPolicy,
        prepare_local_stencils,
    )

    reservation = config.check_capacity(capacity)
    recorder = PhaseRecorder()
    dtype = np.float32 if config.precision == "float32" else np.float64
    points = cloud_points(capacity, config.dimension, seed).astype(dtype)
    neighborhood = recorder.run(
        "search",
        lambda: MeshfreeNeighborhoodPlan(
            points,
            config.neighbors,
            maximum_candidates=config.candidates,
            target_chunk_size=config.chunk_rows,
            precision=MeshfreePrecisionPolicy(geometry_dtype=dtype),
        ).prepare(),
    )
    indices = tuple(
        tuple(2 if axis == d else 0 for axis in range(config.dimension))
        for d in range(config.dimension)
    )
    functional = MeshfreeFunctional(indices, np.ones(config.dimension), name="laplacian")
    prepared = recorder.run(
        "local-fit",
        lambda: prepare_local_stencils(
            neighborhood,
            points,
            points,
            (functional,),
            LocalStencilPolicy(
                polynomial_degree=config.degree, chunk_rows=config.chunk_rows
            ),
        ),
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
    execution = execution_evidence(action, values, config, recorder)
    defect = float(
        np.max(np.abs(np.asarray(execution.pop("result")) - 2 * config.dimension))
    )
    retained = logical_array_bytes((points, prepared))
    recorder.annotate("local-fit", retained_bytes=retained)
    recorder.annotate("warm", polynomial_laplacian_defect=defect)
    if retained > config.working_set_bytes:
        raise DeclaredCapacityRefusal("Actual retained arrays exceed working-set budget.")
    tolerance = 5e-3 if config.precision == "float32" else 1e-7
    if defect > tolerance:
        raise AssertionError(f"Polynomial Laplacian defect {defect} exceeds {tolerance}.")
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "neighbor_capacity": config.neighbors,
        "chunk_rows": config.chunk_rows,
        "storage_capacity": prepared.neighborhood.storage_capacity,
        "domain": "unit-cube-stratified",
        "status": "measured",
        "phases": recorder.record(),
        **execution,
        "retained_bytes": retained,
        "reserved_working_set_bytes": reservation,
        "polynomial_laplacian_defect": defect,
        "refused_rows": prepared.report.refused_rows,
        "prepared_id": prepared.prepared_id,
        "oracle": "independent-host-NumPy: Laplacian(sum(x_i^2))=2d",
    }


def admitted_rows(
    measure: Callable[[int, int, MeshfreeConfig], dict[str, Any]],
    config: MeshfreeConfig,
    /,
) -> list[dict[str, Any]]:
    """Run every requested capacity; refusals and executed failures stay visible."""
    rows: list[dict[str, Any]] = []
    for size in config.sizes:
        for seed in config.seeds:
            with BenchmarkExecutionScope() as execution:
                try:
                    rows.append(measure(size, seed, config))
                except DeclaredCapacityRefusal as error:
                    rows.append(
                        {
                            **refusal_row(size, seed, config, error),
                            **execution.record(),
                            "status": (
                                "post-execution-refusal"
                                if execution.executed
                                else "declared-refusal"
                            ),
                            "failure": failure_record(error),
                        }
                    )
                except Exception as error:  # Verbatim evidence; no retry.
                    rows.append(
                        {
                            "capacity": size,
                            "seed": seed,
                            "dimension": config.dimension,
                            "neighbor_capacity": config.neighbors,
                            "chunk_rows": config.chunk_rows,
                            **execution.record(),
                            "status": "failed",
                            "failure": failure_record(error),
                        }
                    )
    return rows


def observed_slopes(rows: Sequence[dict[str, Any]], key: str, /) -> dict[str, Any]:
    """Least-squares log-log slope of a measured phase over >= 3 measured sizes."""
    measured = sorted(
        (row["capacity"], row["phases"][key]["wall_seconds"])
        for row in rows
        if row.get("status") == "measured" and row["phases"][key]["status"] == "measured"
    )
    if len({capacity for capacity, _ in measured}) < 3:
        return {
            "status": "unavailable",
            "reason": "Fewer than three compatible measured sizes",
        }
    sizes = np.log(np.asarray([capacity for capacity, _ in measured], dtype=np.float64))
    seconds = np.log(np.asarray([value for _, value in measured], dtype=np.float64))
    slope, _ = np.polyfit(sizes, seconds, 1)
    return {
        "status": "measured",
        "phase": key,
        "slope": float(slope),
        "sizes": [capacity for capacity, _ in measured],
        "basis": "wall seconds on a shared loaded host; slope is observed, not a complexity claim",
    }


def make_record(
    config: MeshfreeConfig,
    rows: Sequence[dict[str, Any]],
    driver: Path,
    /,
    *,
    consumers: tuple[Path, ...] = (),
    capacity_variants: tuple[tuple[int, int], ...] = (),
) -> dict[str, Any]:
    configuration = {
        **asdict(config),
        "sizes": list(config.sizes),
        "seeds": list(config.seeds),
    }
    if capacity_variants:
        configuration["capacity_variants"] = [
            {"neighbor_capacity": neighbors, "chunk_rows": chunk}
            for neighbors, chunk in sorted(set(capacity_variants))
        ]
    load = os.getloadavg()
    record: dict[str, Any] = {
        "kind": "meshfree-phase-benchmark",
        "config": configuration,
        "rows": list(rows),
        "environment": capture_environment().to_dict(),
        "precision": config.precision,
        "release_status": "candidate",
        "host_load": {
            "load_average_1_5_15": list(load),
            "logical_cpus": os.cpu_count(),
            "scope": "Shared host; wall time is reported beside process CPU time and compile counts",
        },
        "resource_measurement": {
            "retained_scope": "unique logical payload visible to runtime object walker",
            "compiler_scope": "official executable estimates, not process resident memory",
            "sampled_scope": "per-phase phydrax.execution.PhaseMemorySampler sampled resident peak above phase baseline; sampled, never an upper bound",
        },
        "driver_dependencies": {
            path.relative_to(PROJECT_ROOT).as_posix(): benchmark_driver_fingerprint(
                PROJECT_ROOT, path
            )
            for path in dict.fromkeys((Path(__file__), driver, *consumers))
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
    incompatible = [
        field
        for field in (
            "kind",
            "config",
            "workload_id",
            "environment",
            "capacity_axes",
            "workloads",
        )
        if baseline.get(field) != record.get(field)
    ]
    if incompatible or not isinstance(baseline.get("rows"), list):
        record["performance_before_after"] = {
            "status": "unavailable",
            "baseline_path": str(baseline_path),
            "reason": "Baseline workload/source/environment is incompatible: "
            + ", ".join(incompatible or ["rows"]),
        }
        return
    record["performance_before_after"] = {
        "status": "compared",
        "baseline_path": str(baseline_path),
        "before_identity": baseline["identity"],
        "before": baseline["rows"],
        "after": record["rows"],
    }


def run(
    config: MeshfreeConfig = MeshfreeConfig(),
    /,
    *,
    neighbor_capacities: tuple[int, ...] = (),
    chunk_capacities: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Strong capacity sweep over N and the controlling support/chunk capacities."""
    configure_precision(config, supported=("float64", "float32"))
    effective_variants = tuple(
        sorted(
            {
                (neighbors, chunk)
                for neighbors in neighbor_capacities or (config.neighbors,)
                for chunk in chunk_capacities or (config.chunk_rows,)
            }
        )
    )
    variants = [
        replace(config, neighbors=neighbors, chunk_rows=chunk)
        for neighbors, chunk in effective_variants
    ]
    rows = [
        row for variant in variants for row in admitted_rows(measure_capacity, variant)
    ]
    record = make_record(
        config, rows, Path(__file__), capacity_variants=effective_variants
    )
    record["capacity_axes"] = {
        "sizes": list(config.sizes),
        "neighbor_capacities": [variant.neighbors for variant in variants],
        "chunk_capacities": [variant.chunk_rows for variant in variants],
    }
    record["observed_slopes"] = {
        f"neighbors={variant.neighbors},chunk={variant.chunk_rows}": {
            phase: observed_slopes(
                [
                    row
                    for row in rows
                    if row.get("neighbor_capacity") == variant.neighbors
                    and row.get("chunk_rows") == variant.chunk_rows
                ],
                phase,
            )
            for phase in ("search", "local-fit", "warm", "vjp")
        }
        for variant in variants
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--neighbor-capacities", type=int, nargs="+", default=[])
    parser.add_argument("--chunk-capacities", type=int, nargs="+", default=[])
    args = parser.parse_args()
    record = run(
        config_from_arguments(args),
        neighbor_capacities=tuple(args.neighbor_capacities),
        chunk_capacities=tuple(args.chunk_capacities),
    )
    apply_baseline(record, args.baseline)
    if args.output is not None:
        write_json_atomic(args.output, record)
    else:
        print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

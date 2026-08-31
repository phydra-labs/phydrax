#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class ParticleSplatBenchmarkCase:
    dimension: int
    particle_count: int
    grid_points_per_axis: int
    target_entities: str
    assignment: str
    degree: int
    payload_width: int
    accumulation: str
    route_count: int
    relation_bytes: int
    scalar_workspace_bytes: int
    actual_workspace_bytes: int
    compile_and_first_ms: float
    steady_ms: float
    route_scatter_steady_ms: float | None
    route_scatter_parity_error: float | None
    contributions_per_second: float
    payload_values_per_second: float
    maximum_balance_defect: float
    maximum_partition_defect: float
    maximum_first_moment_defect: float
    maximum_gradient_sum_defect: float
    parity_error: float
    successful: bool


@dataclass(frozen=True)
class ParticleSplatBenchmarkReport:
    maturity: str
    phydrax_version: str
    jax_version: str
    device: str
    cases: tuple[ParticleSplatBenchmarkCase, ...]

    @property
    def passed(self) -> bool:
        return bool(
            self.cases
            and all(case.successful for case in self.cases)
            and max(case.maximum_balance_defect for case in self.cases) < 1e-9
            and max(case.maximum_partition_defect for case in self.cases) < 1e-10
            and max(case.maximum_first_moment_defect for case in self.cases) < 1e-9
            and max(case.maximum_gradient_sum_defect for case in self.cases) < 1e-9
            and max(case.parity_error for case in self.cases) < 1e-9
            and max(
                case.route_scatter_parity_error
                for case in self.cases
                if case.route_scatter_parity_error is not None
            )
            < 1e-9
        )


def _assignment(name: str):
    if name == "multilinear":
        return phx.discretization.MultilinearSplatAssignment(), 1
    if name.startswith("bspline"):
        degree = int(name.removeprefix("bspline"))
        return phx.discretization.TensorBSplineSplatAssignment(degree), degree
    raise ValueError(f"Unknown splat assignment {name!r}.")


def _configuration(
    dimension: int,
    particle_count: int,
    grid_points: int,
    payload_width: int,
    *,
    cell_primary: bool,
):
    axis_type = (
        phx.discretization.UniformCellAxisSpec
        if cell_primary
        else phx.discretization.UniformAxisSpec
    )
    axes = tuple(
        axis_type(
            grid_points,
            periodic=True,
            **({} if cell_primary else {"endpoint": False}),
        )
        for _ in range(dimension)
    )
    names = tuple("xyz"[:dimension])
    grid = phx.discretization.TensorGridPlan(axes, axis_names=names).prepare(
        jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,))))
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count),
        jnp.ones((particle_count,)),
        ambient_dimension=dimension,
    ).prepare()
    keys = jax.random.split(jax.random.key(10 + dimension + payload_width), 2)
    position = jax.random.uniform(keys[0], (particle_count, dimension))
    content = jax.random.normal(keys[1], (particle_count, payload_width))
    return grid, particles, position, content


def _run_case(
    dimension: int,
    particle_count: int,
    grid_points: int,
    assignment_name: str,
    payload_width: int,
    accumulation: str,
    reference: jax.Array | None,
    *,
    cell_primary: bool,
):
    grid, particles, position, content = _configuration(
        dimension,
        particle_count,
        grid_points,
        payload_width,
        cell_primary=cell_primary,
    )
    assignment, degree = _assignment(assignment_name)
    execution = phx.discretization.SplatExecutionPolicy(accumulation=accumulation)
    prepared = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=assignment,
        execution=execution,
    ).prepare(particles)

    @jax.jit
    def apply(current_position, current_content):
        state = prepared.build(current_position)
        return prepared.deposit_content(state, current_content), state

    @jax.jit
    def scatter_routes(current_position, current_content):
        state = prepared.build(current_position)
        payload = state.stencil.weights[..., None] * current_content[:, None, :]
        return prepared.scatter_route_payload(state, payload)

    started = perf_counter()
    first, first_state = apply(position, content)
    jax.block_until_ready(first.content)
    compile_ms = (perf_counter() - started) * 1e3
    repetitions = 5
    started = perf_counter()
    result, state = first, first_state
    for _ in range(repetitions):
        result, state = apply(position, content)
    jax.block_until_ready(result.content)
    steady_ms = (perf_counter() - started) * 1e3 / repetitions
    route_scatter_ms = None
    route_scatter_parity = None
    if accumulation != "compensated":
        scattered = scatter_routes(position, content)
        jax.block_until_ready(scattered.values)
        started = perf_counter()
        for _ in range(repetitions):
            scattered = scatter_routes(position, content)
        jax.block_until_ready(scattered.values)
        route_scatter_ms = (perf_counter() - started) * 1e3 / repetitions
        route_scatter_parity = float(jnp.max(jnp.abs(scattered.values - result.content)))
    parity = (
        0.0 if reference is None else float(jnp.max(jnp.abs(result.content - reference)))
    )
    resources = dict(prepared.preparation.resource_counts)
    route_count = resources["route_count"]
    scalar_workspace = resources["scalar_workspace_bytes"]
    case = ParticleSplatBenchmarkCase(
        dimension=dimension,
        particle_count=particle_count,
        grid_points_per_axis=grid_points,
        target_entities="cell" if cell_primary else "point",
        assignment=assignment_name,
        degree=degree,
        payload_width=payload_width,
        accumulation=accumulation,
        route_count=route_count,
        relation_bytes=resources["relation_bytes"],
        scalar_workspace_bytes=scalar_workspace,
        actual_workspace_bytes=scalar_workspace * payload_width,
        compile_and_first_ms=compile_ms,
        steady_ms=steady_ms,
        contributions_per_second=route_count / (steady_ms * 1e-3),
        payload_values_per_second=(route_count * payload_width) / (steady_ms * 1e-3),
        route_scatter_steady_ms=route_scatter_ms,
        route_scatter_parity_error=route_scatter_parity,
        maximum_balance_defect=float(result.balance.maximum_absolute_balance_defect),
        maximum_partition_defect=float(result.balance.maximum_partition_defect),
        maximum_first_moment_defect=float(
            jnp.max(jnp.abs(state.first_moments), initial=0.0)
        ),
        maximum_gradient_sum_defect=float(
            jnp.max(jnp.abs(state.gradient_sums), initial=0.0)
        ),
        parity_error=parity,
        successful=bool(result.successful),
    )
    return case, result.content


def run_particle_splat_benchmark(*, smoke: bool = False):
    """Benchmark structured particle deposition and its numerical evidence."""
    configurations = (
        (
            (1, 64, 32, "multilinear", 1, False),
            (2, 128, 24, "bspline1", 3, False),
            (2, 128, 24, "bspline2", 8, True),
            (3, 128, 12, "bspline3", 4, True),
        )
        if smoke
        else (
            (1, 4096, 1024, "multilinear", 1, False),
            (2, 8192, 256, "bspline1", 3, False),
            (2, 8192, 256, "bspline2", 8, True),
            (3, 8192, 64, "bspline3", 4, True),
        )
    )
    accumulation_modes = (
        ("fast", "deterministic", "compensated") if smoke else ("fast", "deterministic")
    )
    cases = []
    for (
        dimension,
        particle_count,
        grid_points,
        assignment_name,
        payload_width,
        cell_primary,
    ) in configurations:
        reference = None
        for accumulation in accumulation_modes:
            case, output = _run_case(
                dimension,
                particle_count,
                grid_points,
                assignment_name,
                payload_width,
                accumulation,
                reference,
                cell_primary=cell_primary,
            )
            if reference is None:
                reference = output
            cases.append(case)
    if not smoke:
        compensated, _ = _run_case(
            1,
            64,
            32,
            "multilinear",
            1,
            "compensated",
            None,
            cell_primary=False,
        )
        cases.append(compensated)
    return ParticleSplatBenchmarkReport(
        maturity="experimental",
        phydrax_version=version("phydrax"),
        jax_version=jax.__version__,
        device=str(jax.devices()[0]),
        cases=tuple(cases),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark native structured particle-grid splatting."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_particle_splat_benchmark(smoke=arguments.smoke)
    payload = json.dumps({**asdict(report), "passed": report.passed}, indent=2)
    print(payload)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        temporary.replace(arguments.output)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

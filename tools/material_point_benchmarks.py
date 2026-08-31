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
class MPMBenchmarkCase:
    dimension: int
    particle_count: int
    grid_points_per_axis: int
    accumulation: str
    replay: str
    step_count: int
    route_count: int
    step_workspace_bytes: int
    state_bytes: int
    compile_and_first_step_ms: float
    steady_step_ms: float
    rollout_ms: float
    gradient_ms: float | None
    particle_steps_per_second: float
    route_contributions_per_second: float
    maximum_mass_defect: float
    maximum_momentum_defect: float
    maximum_angular_momentum_defect: float
    replay_parity_error: float
    successful: bool


@dataclass(frozen=True)
class MPMBenchmarkReport:
    maturity: str
    phydrax_version: str
    jax_version: str
    device: str
    cases: tuple[MPMBenchmarkCase, ...]

    @property
    def passed(self) -> bool:
        return bool(
            self.cases
            and all(case.successful for case in self.cases)
            and max(case.maximum_mass_defect for case in self.cases) < 1e-9
            and max(case.maximum_momentum_defect for case in self.cases) < 1e-8
            and max(case.maximum_angular_momentum_defect for case in self.cases) < 1e-8
            and max(case.replay_parity_error for case in self.cases) < 1e-9
        )


def _case(
    dimension,
    particle_count,
    grid_points,
    accumulation,
    replay_mode,
    *,
    measure_gradient=False,
):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(grid_points, periodic=True, endpoint=False)
            for _ in range(dimension)
        ),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))))
    side = int(round(particle_count ** (1.0 / dimension)))
    axes = tuple((jnp.arange(side) + 0.37) / side for _ in range(dimension))
    mesh = jnp.meshgrid(*axes, indexing="ij")
    position = jnp.stack(mesh, axis=-1).reshape((-1, dimension))
    particle_count = int(position.shape[0])
    volume = jnp.full((particle_count,), 1.0 / particle_count)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count), volume, ambient_dimension=dimension
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        execution=phx.discretization.SplatExecutionPolicy(accumulation=accumulation),
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "benchmark",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(dimension),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))),
            periodic=(True,) * dimension,
            support_margin=0.0,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    velocity = 1.0e-3 * jnp.sin(2.0 * jnp.pi * position)
    initial = compiled.initialize_state(position, velocity, volume, arguments)
    dt = 2.0e-4
    step = jax.jit(lambda state: compiled.dynamics.step_detailed(state, dt, arguments))
    started = perf_counter()
    first = step(initial)
    jax.block_until_ready(first.accepted_state.particles.position)
    first_ms = (perf_counter() - started) * 1e3
    repetitions = 5
    started = perf_counter()
    detail = first
    for _ in range(repetitions):
        detail = step(initial)
    jax.block_until_ready(detail.accepted_state.particles.position)
    steady_ms = (perf_counter() - started) * 1e3 / repetitions

    step_count = 4
    temporal = phx.discretization.TemporalMesh.uniform(
        0.0, dt * step_count, step_count, role="internal"
    )
    replay = phx.solver.MPMReplayPolicy(
        replay_mode,
        **({"block_size": 3} if replay_mode == "block" else {}),
    )
    plan = phx.solver.ScheduledMPMRolloutPlan(compiled.dynamics, temporal, replay=replay)
    rollout = jax.jit(lambda state: plan.rollout(state, arguments))
    started = perf_counter()
    result = rollout(initial)
    jax.block_until_ready(result.final_state.particles.position)
    rollout_ms = (perf_counter() - started) * 1e3

    def objective(scale):
        particles_ = phx.discretization.MPMParticleState(
            initial.particles.position,
            initial.particles.velocity * scale,
            initial.particles.deformation_gradient,
            initial.particles.affine_velocity,
            initial.particles.reference_volume,
            initial.particles.first_piola,
            initial.particles.reference_energy_density,
            initial.particles.maximum_wave_speed,
            initial.particles.material_state,
        )
        state = phx.discretization.MPMRuntimeState(
            particles_, initial.time, initial.accepted_step, initial.last_status
        )
        return jnp.sum(plan.rollout(state, arguments).final_state.particles.position ** 2)

    gradient_ms = None
    if measure_gradient:
        gradient = jax.jit(jax.grad(objective))
        started = perf_counter()
        gradient_value = gradient(jnp.asarray(1.0))
        jax.block_until_ready(gradient_value)
        gradient_ms = (perf_counter() - started) * 1e3

    reference = phx.solver.ScheduledMPMRolloutPlan(
        compiled.dynamics,
        temporal,
        replay=phx.solver.MPMReplayPolicy("full"),
    ).rollout(initial, arguments)
    parity = float(
        jnp.max(
            jnp.abs(
                result.final_state.particles.position
                - reference.final_state.particles.position
            )
        )
    )
    resources = dict(compiled.dynamics.preparation.resource_counts)
    route_count = int(resources["route_count"])
    return MPMBenchmarkCase(
        dimension,
        particle_count,
        grid_points,
        accumulation,
        replay_mode,
        step_count,
        route_count,
        int(resources["step_workspace_bytes"]),
        int(resources["state_bytes"]),
        first_ms,
        steady_ms,
        rollout_ms,
        gradient_ms,
        particle_count / (steady_ms * 1e-3),
        route_count / (steady_ms * 1e-3),
        float(detail.diagnostics.transfer.relative_mass_defect),
        float(detail.diagnostics.transfer.relative_momentum_defect),
        float(detail.diagnostics.transfer.relative_angular_momentum_defect),
        parity,
        bool(detail.successful) and bool(jnp.all(result.accepted)),
    )


def run_material_point_benchmark(*, smoke=False):
    configurations = ((2, 64, 16), (3, 27, 8)) if smoke else ((2, 576, 48), (3, 216, 18))
    cases = []
    for dimension, particles, grid in configurations:
        cases.append(_case(dimension, particles, grid, "fast", "full"))
        cases.append(
            _case(
                dimension,
                particles,
                grid,
                "deterministic",
                "block",
                measure_gradient=dimension == 2,
            )
        )
    cases.append(
        _case(
            2,
            configurations[0][1],
            configurations[0][2],
            "deterministic",
            "full",
        )
    )
    return MPMBenchmarkReport(
        "experimental",
        version("phydrax"),
        jax.__version__,
        str(jax.devices()[0]),
        tuple(cases),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark explicit APIC MPM.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/material_point.json"),
    )
    arguments = parser.parse_args()
    report = run_material_point_benchmark(smoke=arguments.smoke)
    payload = json.dumps({**asdict(report), "passed": report.passed}, indent=2)
    print(payload)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(payload + "\n")
    temporary.replace(arguments.output)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

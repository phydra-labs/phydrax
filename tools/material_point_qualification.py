#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


@dataclass(frozen=True)
class MPMQualificationCase:
    name: str
    dimension: int
    particles: int
    grid_nodes: int
    steps: int
    relative_error: float
    maximum_mass_defect: float
    maximum_momentum_defect: float
    maximum_angular_momentum_defect: float
    maximum_absolute_energy_defect: float
    minimum_jacobian: float
    maximum_apic_condition: float
    accepted: bool
    passed: bool


@dataclass(frozen=True)
class MPMQualificationReport:
    maturity: str
    phydrax_version: str
    jax_version: str
    device: str
    wave_convergence_order: float
    cases: tuple[MPMQualificationCase, ...]

    @property
    def passed(self) -> bool:
        return bool(
            self.cases
            and all(case.passed for case in self.cases)
            and self.wave_convergence_order >= 0.75
        )


def _compile_periodic(
    dimension: int,
    grid_shape: tuple[int, ...],
    bounds: jax.Array,
    position: jax.Array,
    volume: jax.Array,
):
    axes = tuple(
        phx.discretization.UniformAxisSpec(size, periodic=True, endpoint=False)
        for size in grid_shape
    )
    grid = phx.discretization.TensorGridPlan(
        axes, axis_names=tuple("xyz"[:dimension])
    ).prepare(bounds)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0]),
        volume,
        ambient_dimension=dimension,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(particles)
    material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(dimension)
    problem = phx.equations.MaterialPointProblemIR("qualification", material)
    compiled = phx.equations.compile_material_point_problem(
        problem,
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            bounds,
            periodic=(True,) * dimension,
            support_margin=0.0,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    return compiled, arguments


def _case_from_rollout(name, dimension, compiled, result, error):
    mass = float(jnp.nanmax(result.relative_mass_defects))
    momentum = float(jnp.nanmax(result.relative_momentum_defects))
    angular = float(jnp.nanmax(result.relative_angular_momentum_defects))
    energy = float(jnp.nanmax(jnp.abs(result.energy_balance_defects)))
    minimum_j = float(jnp.nanmin(result.minimum_jacobians))
    condition = float(jnp.nanmax(result.maximum_apic_conditions))
    accepted = bool(jnp.all(result.accepted))
    passed = bool(
        accepted
        and mass < 1.0e-10
        and momentum < 1.0e-9
        and angular < 1.0e-9
        and minimum_j > 0.0
        and condition <= 10.0
        and np.isfinite(error)
    )
    return MPMQualificationCase(
        name,
        dimension,
        compiled.dynamics.particles.capacity,
        compiled.dynamics.splat.target_size,
        int(result.accepted.shape[0]),
        float(error),
        mass,
        momentum,
        angular,
        energy,
        minimum_j,
        condition,
        accepted,
        passed,
    )


def _wave_case(resolution: int):
    ny = max(4, resolution // 4)
    x = (jnp.arange(resolution) + 0.37) / resolution
    y = (jnp.arange(ny) + 0.37) * (0.25 / ny)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    position = jnp.stack((xx, yy), axis=-1).reshape((-1, 2))
    particle_volume = (1.0 / resolution) * (0.25 / ny)
    volume = jnp.full((position.shape[0],), particle_volume)
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 0.25]])
    compiled, arguments = _compile_periodic(2, (resolution, ny), bounds, position, volume)
    amplitude = 1.0e-3
    wave_number = 2.0 * jnp.pi
    speed = jnp.sqrt(8.0 + 4.0 * 2.0 / 3.0)
    velocity = jnp.stack(
        (
            amplitude * jnp.sin(wave_number * position[:, 0]),
            jnp.zeros(position.shape[0]),
        ),
        axis=-1,
    )
    gradient = amplitude * wave_number * jnp.cos(wave_number * position[:, 0])
    affine = jnp.zeros((position.shape[0], 2, 2)).at[:, 0, 0].set(gradient)
    final_time_target = 0.015 / speed
    dt_target = 0.15 * min(1.0 / resolution, 0.25 / ny) / speed
    steps = max(2, int(np.ceil(float(final_time_target / dt_target))))
    final_time = steps * dt_target
    initial = compiled.initialize_state(
        position,
        velocity,
        volume,
        arguments,
        affine_velocity=affine,
    )
    mesh = phx.discretization.TemporalMesh.uniform(
        0.0, float(final_time), steps, role="internal"
    )
    result = phx.solver.ScheduledMPMRolloutPlan(
        compiled.dynamics,
        mesh,
        replay=phx.solver.MPMReplayPolicy("block", block_size=max(2, steps - 1)),
    ).rollout(initial, arguments)
    expected_velocity = (
        amplitude
        * jnp.sin(wave_number * position[:, 0])
        * jnp.cos(speed * wave_number * final_time)
    )
    error = (
        jnp.sqrt(
            jnp.mean(
                (result.final_state.particles.velocity[:, 0] - expected_velocity) ** 2
            )
        )
        / amplitude
    )
    return _case_from_rollout(
        f"plane-strain-wave-{resolution}", 2, compiled, result, error
    )


def _translation_case(dimension: int):
    position = jnp.asarray(
        [
            [0.23, 0.27, 0.31],
            [0.39, 0.34, 0.45],
            [0.32, 0.48, 0.58],
            [0.47, 0.51, 0.63],
        ]
    )[:, :dimension]
    bounds = jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,))))
    volume = jnp.full((4,), 0.01)
    compiled, arguments = _compile_periodic(
        dimension, (10,) * dimension, bounds, position, volume
    )
    prescribed = jnp.asarray((0.04, -0.02, 0.03))[:dimension]
    velocity = jnp.broadcast_to(prescribed, position.shape)
    initial = compiled.initialize_state(position, velocity, volume, arguments)
    mesh = phx.discretization.TemporalMesh.uniform(0.0, 0.004, 4, role="internal")
    result = phx.solver.ScheduledMPMRolloutPlan(
        compiled.dynamics,
        mesh,
        replay=phx.solver.MPMReplayPolicy("block", block_size=3),
    ).rollout(initial, arguments)
    expected = position + 0.004 * velocity
    error = jnp.max(jnp.abs(result.final_state.particles.position - expected))
    return _case_from_rollout(
        f"translation-{dimension}d", dimension, compiled, result, error
    )


def _rollback_case():
    position = jnp.asarray([[0.23, 0.27], [0.39, 0.34], [0.32, 0.48], [0.47, 0.51]])
    volume = jnp.full((4,), 0.01)
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    compiled, arguments = _compile_periodic(2, (10, 10), bounds, position, volume)
    initial = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    detail = compiled.dynamics.step_detailed(initial, 10.0, arguments)
    exact = all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(
            jax.tree.leaves(initial.particles),
            jax.tree.leaves(detail.accepted_state.particles),
            strict=True,
        )
    )
    passed = (
        not bool(detail.successful)
        and exact
        and int(detail.accepted_state.accepted_step) == 0
        and float(detail.accepted_state.time) == float(initial.time)
    )
    return MPMQualificationCase(
        "oversized-step-rollback",
        2,
        compiled.dynamics.particles.capacity,
        compiled.dynamics.splat.target_size,
        1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        False,
        passed,
    )


def run_material_point_qualification(*, smoke: bool = False):
    resolutions = (8, 12) if smoke else (12, 24, 36)
    wave_cases = tuple(_wave_case(value) for value in resolutions)
    errors = np.asarray([case.relative_error for case in wave_cases])
    widths = 1.0 / np.asarray(resolutions, dtype=float)
    if errors.size < 2 or np.any(errors <= 0.0):
        order = 0.0
    else:
        order = float(np.polyfit(np.log(widths), np.log(errors), deg=1)[0])
    cases = wave_cases + (
        _translation_case(2),
        _translation_case(3),
        _rollback_case(),
    )
    return MPMQualificationReport(
        "experimental",
        version("phydrax"),
        jax.__version__,
        str(jax.devices()[0]),
        order,
        cases,
    )


def main():
    parser = argparse.ArgumentParser(description="Qualify explicit APIC material points.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/material_point_qualification.json"),
    )
    arguments = parser.parse_args()
    report = run_material_point_qualification(smoke=arguments.smoke)
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

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Route, transfer, and explicit-step timings; compilation is reported separately.

python -m benchmarks.mpm_volumetric_stabilization --particles 64 1024 16384
Every full-step sample starts from the same committed state (no time-drift bias).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax.discretization.mpm import (
    build_apic_route_payload,
    gather_apic,
    MPMVolumetricStabilizationKind,
    MPMVolumetricStabilizationPlan,
)


_T = TypeVar("_T")
_R = TypeVar("_R")


def _invoke(operation: Callable[[_T], _R], argument: _T) -> _R:
    return operation(argument)


_COMPILED_INVOKE = eqx.filter_jit(_invoke)


def _phase(
    operation: Callable[[_T], _R], argument: _T, repeats: int
) -> tuple[_R, dict[str, object]]:
    # Equinox erases the documented lowering interface from its annotation.
    lowering = _COMPILED_INVOKE.lower  # ty: ignore[unresolved-attribute]
    executable, compilation = measure_lower_and_compile(
        lambda: lowering(operation, argument), lambda lowered: lowered.compile()
    )
    result, distribution = measure_repeated(
        lambda: executable(operation, argument), warmup=2, repeats=repeats
    )
    evidence = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Backend does not expose all compiler estimates.",
    )
    return result, {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "steady_seconds": distribution.to_dict(unit="seconds"),
        "compiler": asdict(evidence),
        "logical_output_bytes": logical_array_bytes(result),
    }


def benchmark_case(
    count: int, kind: MPMVolumetricStabilizationKind, repeats: int
) -> dict[str, object]:
    if count < 1:
        raise ValueError("particle counts must be positive.")
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(65, periodic=False, endpoint=True),) * 2,
        axis_names=("x", "y"),
    ).prepare(bounds)
    position = jnp.asarray(
        np.random.default_rng(4901).uniform(0.25, 0.75, size=(count, 2)),
        dtype=jnp.float64,
    )
    volume = jnp.full((count,), 0.25 / count, dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count, dtype=jnp.int32), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        boundary="reject",
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "volumetric-benchmark",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            volumetric_stabilization=MPMVolumetricStabilizationPlan(kind)
        ),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.05, 0.05], [0.95, 0.95]], dtype=jnp.float64),
            periodic=(False, False),
            support_margin=0.04,
        ),
    )
    dynamics = compiled.dynamics
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 1000.0)
    )
    velocity = 0.01 * (position - 0.5)
    state = compiled.initialize_state(position, velocity, volume, arguments)
    routes, route_timing = _phase(dynamics._build_routes, position, repeats)
    active = jnp.ones((count,), dtype=jnp.bool_)

    def transfer(
        value: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        particle_velocity, piola = value
        payload = build_apic_route_payload(
            routes,
            volume,
            particle_velocity,
            state.particles.affine_velocity,
            volume,
            piola,
            state.particles.deformation_gradient,
            jnp.zeros_like(particle_velocity),
            active,
            physical_deformation_gradient=state.particles.physical_deformation_gradient,
        )
        mass_payload = routes.stencil.weights * volume[:, None]
        scattered = splat.scatter_route_payload(
            routes, jnp.concatenate((mass_payload[..., None], payload), axis=-1)
        )
        mass = scattered.values[..., 0]
        node_active = mass > 0.0
        nodal_velocity = jnp.where(
            node_active[..., None],
            scattered.values[..., 1:3] / jnp.where(node_active, mass, 1.0)[..., None],
            0.0,
        )
        gathered = gather_apic(routes, nodal_velocity.reshape(-1, 2), active, 1e12)
        updated = dynamics._update_deformation(
            routes,
            state.particles.deformation_gradient,
            gathered.velocity_gradient,
            jnp.asarray(1e-5, dtype=jnp.float64),
            volume,
            active,
            reference_volume=volume,
            physical_deformation_gradient=state.particles.physical_deformation_gradient,
        )
        return updated, scattered.values, gathered.successful & scattered.successful

    transfer_result, transfer_timing = _phase(
        transfer, (velocity, state.particles.first_piola), repeats
    )
    deformation, _, transfer_successful = transfer_result
    detail, step_timing = _phase(
        lambda value: dynamics.step_detailed(
            value, jnp.asarray(1e-5, dtype=jnp.float64), arguments
        ),
        state,
        repeats,
    )
    return {
        "mode": kind,
        "particle_count": count,
        "route_count_per_particle": routes.stencil.indices.shape[1],
        "phases": {
            "route_build": route_timing,
            "transfer": transfer_timing,
            "full_step": step_timing,
        },
        "logical_state_bytes": logical_array_bytes(state),
        "successful": bool(
            routes.successful
            & detail.successful
            & transfer_successful
            & jnp.all(jnp.isfinite(deformation))
        ),
        "relative_mass_defect": float(detail.diagnostics.transfer.relative_mass_defect),
        "relative_momentum_defect": float(
            detail.diagnostics.transfer.relative_momentum_defect
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, nargs="+", default=(64, 1024, 16384))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=str)
    parser.add_argument(
        "--contended",
        action="store_true",
        help="Record concurrent scientific workloads honestly.",
    )
    arguments = parser.parse_args()
    modes: tuple[MPMVolumetricStabilizationKind, ...] = ("none", "b_bar", "f_bar")
    cases = [
        benchmark_case(count, kind, arguments.repeats)
        for count in arguments.particles
        for kind in modes
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "contended": arguments.contended,
        "cases": cases,
        "passed": all(case["successful"] for case in cases),
    }
    if arguments.output:
        write_json_atomic(arguments.output, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

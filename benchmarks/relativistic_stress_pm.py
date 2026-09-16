from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from benchmarks._runtime import capture_environment, logical_array_bytes, measure_repeated
from phydrax._physical import RelativityScaleContract
from phydrax.applications.cosmology._scales import CODE_COSMOLOGY_SCALE
from phydrax.applications.cosmology._weak_field_relativistic_pm import (
    WeakFieldRelativisticPMPlan,
    WeakFieldRelativisticPMPolicy,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.discretization import AxisDomain, FourierBasisPlan, TensorSpectralPlan
from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._relativistic_stress_transfer import (
    RelativisticStressDepositPlan,
)
from phydrax.discretization.splatting import ParticleGridSplatPlan
from phydrax.metrix import ADMGridGeometry, RelativityConvention


def _build(count: int):
    convention = RelativityConvention.canonical()
    scale = RelativityScaleContract(CODE_COSMOLOGY_SCALE, 1, 1, 1, 1)
    units = RelativisticUnitContract(scale, convention)
    spectral = TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="weak-field-benchmark",
    ).prepare(tuple(AxisDomain.periodic(0.0, 1.0) for _ in range(3)))
    positions = spectral.grid.points
    capacity = positions.shape[0]
    particles = ParticleSetPlan(
        jnp.arange(capacity),
        jnp.ones((capacity,)),
        ambient_dimension=3,
    ).prepare()
    transfer = ParticleGridSplatPlan(spectral.grid).prepare(particles)
    stress = RelativisticStressDepositPlan(
        transfer,
        units,
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([1.0]),
        mass_shell_relative_tolerance=1e-5,
        conservation_tolerance=1e-5,
        frame_momentum_relative_tolerance=1e-5,
    )
    policy = WeakFieldRelativisticPMPolicy(
        maximum_scalar_metric_fraction=0.2,
        maximum_vector_metric_fraction=0.2,
        maximum_tensor_metric_norm=0.2,
        maximum_cell_crossing=0.5,
        constraint_relative_tolerance=5e-4,
        gauge_absolute_tolerance=5e-4,
        force_relative_tolerance=5e-4,
        conservation_relative_tolerance=5e-3,
    )
    plan = WeakFieldRelativisticPMPlan(
        stress,
        spectral,
        units,
        gravitational_constant=1e-4,
        policy=policy,
    )
    shape = spectral.physical_shape
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    geometry = ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        identity,
        identity,
        jnp.ones(shape),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        snapshot_token=jnp.asarray(1, dtype=jnp.int32),
        chart_id="periodic-cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id=spectral.grid.topology.topology_id,
        geometry_lineage_id="relativistic-stress-pm-benchmark",
    )
    xyz = positions.reshape(shape + (3,))
    coordinates = jnp.concatenate((jnp.zeros(shape + (1,)), xyz), axis=-1)
    frame = LocalRelativisticFramePlan.from_adm(
        geometry,
        units,
        coordinates,
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="eulerian-observer",
        orientation_id="cartesian-right-handed",
    )
    phase = 2.0 * jnp.pi * positions[:, 0]
    weights = jnp.ones((capacity,))
    momenta = jnp.stack(
        (
            0.15 * jnp.sin(phase),
            0.30 * jnp.cos(phase),
            0.12 * jnp.sin(phase),
        ),
        axis=-1,
    )
    state = stress.initialize(
        weights,
        positions,
        momenta,
        jnp.ones((capacity,), dtype=jnp.int32),
        frame,
    )
    return plan, frame, state


def main() -> int:
    count = 6
    plan, frame, state = _build(count)
    operation = jax.jit(lambda current: plan.solve_stress(current, frame))
    result, timing = measure_repeated(lambda: operation(state), warmup=1, repeats=3)
    payload = {
        "benchmark": "relativistic-stress-poisson-gauge-pm",
        "grid_shape": list(plan.spectral.physical_shape),
        "particle_capacity": state.capacity,
        "sectors": {
            "phi": True,
            "psi": True,
            "transverse_vector": True,
            "transverse_traceless_tensor": True,
            "scalar_only": plan.scalar_only,
        },
        "residuals": {
            "scalar": float(result.scalar_residual),
            "vector": float(result.vector_residual),
            "tensor": float(result.tensor_residual),
            "gauge": float(result.gauge_defect),
            "force": float(result.force_relative_defect),
            "omitted_channel_bound": float(result.omitted_channel_bound),
            "spectral_support": float(result.spectral_support_defect),
        },
        "resources": {
            "grid_points": plan.grid_points,
            "particle_routes": plan.stress_transfer.transfer.route_count,
            "workspace_bytes": plan.workspace_bytes,
            "logical_result_bytes": logical_array_bytes(result),
        },
        "timing": timing.to_milliseconds_dict(),
        "successful": bool(result.successful),
        "environment": capture_environment().to_dict(),
        "release_claim": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

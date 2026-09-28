import jax.numpy as jnp

from phydrax.applications.foams import (
    FoamDynamicsPlan,
    FoamDynamicsState,
    FoamMaterialPlan,
    PreparedFoamDynamics,
    RegionPressureAirPlan,
    VortexSheetAirPlan,
)
from phydrax.geometry.multiregion_surface import PreparedMultiRegionSurface, seed_sphere


def test_perturbed_bubble_vortex_air_workflow_closes_volume_and_fmm_evidence() -> None:
    seed = seed_sphere(1.0, subdivisions=0)
    topology = seed.topology(seed.capacity_plan(resource_id="vortex-air-workflow"))
    base = seed.state(topology)
    radius = jnp.linalg.norm(base.positions, axis=1)
    direction = jnp.where(
        radius[:, None] > 0.0,
        base.positions / jnp.maximum(radius[:, None], jnp.finfo(radius.dtype).tiny),
        0.0,
    )
    cosine = direction[:, 2]
    mode = 0.5 * (3.0 * cosine * cosine - 1.0)
    positions = base.positions * (1.0 + 1.0e-3 * mode)[:, None]
    surface_state = base.with_positions(positions)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    target = surface.region_volumes(positions)[
        jnp.asarray(topology.finite_region_indices, dtype=jnp.int32)
    ]
    dynamics_state = FoamDynamicsState(surface_state)
    dynamics = PreparedFoamDynamics(
        FoamDynamicsPlan(
            route="film-inertia",
            time_step=1.0e-6,
            volume_tolerance=1.0e-10,
        ),
        surface,
        FoamMaterialPlan.soap_film(topology.region_ids, 0.025),
        RegionPressureAirPlan.incompressible(target),
        dynamics_state,
    )
    prepared = VortexSheetAirPlan(
        air_density=1.2,
        time_step=1.0e-6,
        fmm_depth=1,
        fmm_leaf_capacity=64,
        maximum_fmm_relative_error=0.15,
    ).prepare(dynamics, dynamics_state)

    result = prepared.advance(prepared.initialize(dynamics_state))

    assert result.successful
    radial_velocity = jnp.sum(result.state.surface.velocities * direction, axis=1)
    restoring_mode_velocity = jnp.sum(
        jnp.where(topology.vertex_active, radial_velocity * mode, 0.0)
    )
    assert float(restoring_mode_velocity) < 0.0
    assert float(result.evidence.volume_residual) < 1.0e-10
    assert bool(result.evidence.fmm_successful)
    assert bool(result.evidence.direct_reference_evaluated)
    assert float(result.evidence.fmm_relative_l2_error) < 0.15
    assert result.evidence.core_policy == "mean-edge-fraction"
    assert result.evidence.source_epoch == result.evidence.target_epoch

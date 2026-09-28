"""Accepted film-thickness rupture followed by constrained foam relaxation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.applications.foams import (
    apply_foam_rupture,
    FoamDynamicsPlan,
    FoamDynamicsState,
    FoamMaterialPlan,
    FoamRupturePlan,
    PreparedFoamDynamics,
    RegionPressureAirPlan,
)
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceState,
    PreparedMultiRegionSurface,
    seed_double_bubble,
)
from phydrax.interfacial_transport import FilmStepStatus, SurfaceFilmEvidence


def run() -> dict[str, object]:
    seed = seed_double_bubble(1.0, 0.8, ring_points=12)
    topology = seed.topology(seed.capacity_plan(resource_id="advanced-foam-burst"))
    base = seed.state(topology)
    sheet_fields = np.zeros(
        (topology.vertex_capacity, topology.slot_width, 2), dtype=np.float64
    )
    sheet_fields[np.asarray(topology.slot_active), 0] = 2.0e-10
    sheet_fields[np.asarray(topology.slot_active), 1] = 1.0e-6
    region_fields = np.zeros((topology.region_capacity, 2), dtype=np.float64)
    region_fields[: topology.region_count, 0] = (1.0, 0.8, 0.0)
    region_fields[: topology.region_count, 1] = (3.0, 2.4, 0.0)
    state = MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=sheet_fields,
        region_fields=region_fields,
        sheet_field_names=("film_liquid_volume", "circulation"),
        region_field_names=("gas_amount_mol", "gas_internal_energy_j"),
    )
    finite = np.flatnonzero(np.asarray(topology.region_finite))
    pairs = np.asarray(topology.region_pairs[: topology.region_pair_count])
    separating_pair = int(
        np.flatnonzero(np.all(pairs == np.sort(finite)[None, :], axis=1))[0]
    )
    pair_slots = np.asarray(topology.vertex_pair_slots) == separating_pair
    thickness = np.full(
        (topology.vertex_capacity, topology.slot_width), 8.0e-7, dtype=np.float64
    )
    thickness[pair_slots] = 4.0e-8
    film_evidence = SurfaceFilmEvidence(
        liquid_volume_residual_m3=jnp.asarray(0.0),
        boundary_exchange_m3=jnp.asarray(0.0),
        minimum_thickness_m=jnp.asarray(4.0e-8),
        rupture_mask=jnp.asarray(pair_slots),
        energy_change_j=jnp.asarray(-1.0e-12),
        dissipation_guaranteed=jnp.asarray(True),
        positivity_guaranteed=jnp.asarray(True),
        conductance_admissible=jnp.asarray(True),
        nonlinear_status=jnp.asarray(0, dtype=jnp.int32),
        nonlinear_iterations=jnp.asarray(5, dtype=jnp.int32),
        nonlinear_residual_norm=jnp.asarray(1.0e-12),
        converged=jnp.asarray(True),
        finite=jnp.asarray(True),
        geometry_revision=jnp.asarray(7, dtype=jnp.int32),
    )
    rupture = apply_foam_rupture(
        FoamRupturePlan(4.0e-8, minimum_trigger_slots=2),
        topology,
        state,
        thickness,
        FilmStepStatus.ACCEPTED,
        film_evidence,
        7,
        0.0,
    )
    if not rupture.successful:
        return {
            "rupture_status": int(rupture.evidence.status),
            "committed": False,
        }
    surface = PreparedMultiRegionSurface(rupture.topology, rupture.state)
    volumes = surface.region_volumes(rupture.state.positions)[
        jnp.asarray(rupture.topology.finite_region_indices, dtype=jnp.int32)
    ]
    air = RegionPressureAirPlan.incompressible(volumes)
    dynamics_state = FoamDynamicsState(
        rupture.state,
        unresolved_rim_content=rupture.unresolved_rim_content,
    )
    dynamics = PreparedFoamDynamics(
        FoamDynamicsPlan(
            route="overdamped",
            time_step=1.0e-4,
            friction=5.0,
        ),
        surface,
        FoamMaterialPlan.soap_film(rupture.topology.region_ids, 0.025),
        air,
        dynamics_state,
    ).advance(dynamics_state)
    return {
        "rupture_status": int(rupture.evidence.status),
        "committed": rupture.successful,
        "source_regions": topology.region_count,
        "target_regions": rupture.topology.region_count,
        "unresolved_rim_content": float(rupture.unresolved_rim_content),
        "liquid_residual": float(rupture.evidence.liquid_conservation_residual),
        "gas_amount_residual": float(rupture.evidence.gas_amount_residual),
        "gas_energy_residual": float(rupture.evidence.gas_energy_residual),
        "region_lineage": rupture.evidence.gas_region_lineage,
        "dynamics_status": int(dynamics.evidence.status),
        "volume_residual": float(dynamics.evidence.volume_residual),
        "derivative_available": bool(dynamics.evidence.derivative_available),
    }


if __name__ == "__main__":
    print(run())

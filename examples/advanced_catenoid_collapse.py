"""Catenoid soap film continued to its stability limit and collapsed past it.

Rings of radius R at z = +/- d (Goldstein et al., Phys. Rev. E 104, 035105,
2021): the stable catenoid exists for D = d / R < D_c = 0.6627. Each branch
step solves the discrete minimal surface on a fresh band seeded with the exact
stable-branch profile and compares neck radius and area with the continuum.

Past the limit no catenoid exists. The band is started at D = 0.7 from the
critical profile and driven by overdamped, inertia-free capillary relaxation
(film friction only; no air or film inertia). Between relaxation calls one
transactional event pass remeshes the shrinking neck (collapses, splits,
flips) and, once the neck is a three-edge loop shorter than the declared neck
perimeter, pinches it and splits the inner label into two children with
lineage. The two resulting disks then relax toward the flat discs spanning
each ring (area pi R^2 each). Film liquid volume is carried as an extensive
sheet field and must be conserved exactly through every event.
"""

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from phydrax.applications.foams import (
    catenoid_area_ratio,
    catenoid_critical_parameters,
    catenoid_stable_neck_ratio,
    FoamEquilibriumPlan,
    FoamEquilibriumStatus,
    FoamMaterialPlan,
    FoamRelaxationPlan,
    FoamWireConstraints,
    PreparedFoamEquilibrium,
    PreparedFoamRelaxation,
)
from phydrax.geometry.multiregion_surface import (
    apply_surface_events,
    multiregion_sheet_views,
    MultiRegionRemeshPlan,
    MultiRegionSurfaceState,
    PreparedMultiRegionSurface,
    propose_pinches,
    propose_remesh,
    RegionSplitProposal,
    seed_catenoid,
    SurfaceEventKind,
    SurfaceEventPolicy,
)


SURFACE_TENSION = 0.025
FILM_THICKNESS = 2.0e-6


def _solve(ratio: float, neck: float) -> tuple[bool, float, float]:
    seed = seed_catenoid(1.0, ratio, ring_points=24, rows=10, neck_radius=neck)
    topology = seed.topology(seed.capacity_plan(resource_id="catenoid-example"))
    state = seed.state(topology)
    rings = seed.vertex_set("ring-lower") + seed.vertex_set("ring-upper")
    wires = FoamWireConstraints(rings, seed.positions[seed.vertex_indices(rings)])
    material = FoamMaterialPlan.soap_film(
        topology.region_ids, SURFACE_TENSION, wires=wires
    )
    surface = PreparedMultiRegionSurface(topology, state)
    equilibrium = PreparedFoamEquilibrium(FoamEquilibriumPlan(), surface, material, state)
    result = equilibrium.solve(state, equilibrium.parameters(jnp.zeros((0,))))
    points = np.asarray(result.state.positions[: topology.vertex_count])
    converged = int(result.evidence.status) == FoamEquilibriumStatus.CONVERGED
    area = float(result.energy) / (2.0 * SURFACE_TENSION) / (2.0 * np.pi)
    return converged, float(np.min(np.linalg.norm(points[:, :2], axis=1))), area


def _collapse(ratio: float, neck: float, epochs: int = 60) -> dict[str, Any]:
    """Overdamped collapse past the limit with remeshing, pinch and region split."""
    seed = seed_catenoid(1.0, ratio, ring_points=16, rows=10, neck_radius=neck)
    topology = seed.topology(
        seed.capacity_plan(
            resource_id="catenoid-collapse", headroom=3.0, event_capacity=64
        )
    )
    base = seed.state(topology)
    slot_area = PreparedMultiRegionSurface(topology, base).slot_areas(base.positions)
    state = MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=FILM_THICKNESS * np.asarray(slot_area)[:, :, None],
        sheet_field_names=("liquid",),
    )
    liquid = float(jnp.sum(state.sheet_fields))
    rings = seed.vertex_set("ring-lower") + seed.vertex_set("ring-upper")
    wires = FoamWireConstraints(rings, seed.positions[seed.vertex_indices(rings)])
    relaxation = FoamRelaxationPlan(friction=1.0, time_step=1.0, steps=40)
    remesh = MultiRegionRemeshPlan(
        minimum_edge_length=0.06,
        maximum_edge_length=0.45,
        minimum_angle=math.radians(15.0),
    )
    policy = SurfaceEventPolicy(fixed_vertex_ids=rings)
    elapsed, pinch_time, lineage, energy = 0.0, None, None, math.nan
    relaxation_status, event_pass_status, pinch_epoch = None, None, None
    events = {kind.name: 0 for kind in SurfaceEventKind}
    disk_relaxed = False
    polygon_disk_area = 0.5 * 16.0 * math.sin(2.0 * math.pi / 16.0)
    completed_epochs = 0
    for epoch in range(epochs):
        surface = PreparedMultiRegionSurface(topology, state)
        material = FoamMaterialPlan.soap_film(
            topology.region_ids, SURFACE_TENSION, wires=wires
        )
        relaxed = PreparedFoamRelaxation(relaxation, surface, material, state).relax(
            state, jnp.zeros((0,))
        )
        relaxation_status = int(relaxed.evidence.status)
        completed_epochs = epoch + 1
        if not relaxed.successful:
            break
        energy = float(relaxed.energy)
        elapsed += float(relaxed.evidence.elapsed_time)
        state = relaxed.state
        surface = PreparedMultiRegionSurface(topology, state)
        necks = propose_pinches(surface, state, maximum_neck_perimeter=0.25)
        proposals = [*necks, *propose_remesh(surface, state, remesh)]
        if necks:
            proposals.append(RegionSplitProposal("core"))
        result = apply_surface_events(topology, state, proposals, policy=policy)
        event_pass_status = int(result.evidence.status)
        for record in result.evidence.records:
            events[record.kind.name] += int(record.accepted)
        if result.committed and any(
            r.kind is SurfaceEventKind.PINCH and r.accepted
            for r in result.evidence.records
        ):
            pinch_time, lineage, pinch_epoch = (
                elapsed,
                result.evidence.lineage,
                epoch,
            )
        topology, state = result.topology, result.state
        if pinch_epoch is not None and epoch > pinch_epoch:
            current_points = np.asarray(state.positions[: topology.vertex_count])
            current_surface = PreparedMultiRegionSurface(topology, state)
            current_disk_area = (
                float(jnp.sum(current_surface.face_areas(state.positions))) / 2.0
            )
            current_flatness = float(np.max(np.abs(np.abs(current_points[:, 2]) - ratio)))
            disk_relaxed = (
                abs(current_disk_area / polygon_disk_area - 1.0) < 1.0e-3
                and current_flatness < 1.0e-2
            )
            if disk_relaxed:
                break
    points = np.asarray(state.positions[: topology.vertex_count])
    surface = PreparedMultiRegionSurface(topology, state)
    views = multiregion_sheet_views(surface, state)
    disk_area = float(jnp.sum(surface.face_areas(state.positions))) / 2.0
    successful = (
        pinch_time is not None
        and disk_relaxed
        and surface.evidence.accepted
        and relaxation_status == 0
    )
    return {
        "successful": successful,
        "D": ratio,
        "pinched": pinch_time is not None,
        "pinch_time": pinch_time,
        "disk_relaxed": disk_relaxed,
        "completed_epochs": completed_epochs,
        "relaxation_status": relaxation_status,
        "event_pass_status": event_pass_status,
        "validation_status": int(surface.evidence.status),
        "validation_accepted": surface.evidence.accepted,
        "region_ids": topology.region_ids,
        "region_lineage": None if lineage is None else lineage.region_parents,
        "pinch_vertex_parents": None if lineage is None else lineage.vertex_parents[-2:],
        "disk_euler_characteristics": [
            view.mesh.topology.euler_characteristic for view in views.views
        ],
        "disk_area_over_pi_R2": disk_area / math.pi,
        "disk_area_over_polygon": disk_area / polygon_disk_area,
        "flatness": float(np.max(np.abs(np.abs(points[:, 2]) - ratio))),
        "accepted_events": events,
        "liquid_relative_defect": abs(float(jnp.sum(state.sheet_fields)) - liquid)
        / liquid,
        "final_energy_over_disks": energy / (2.0 * SURFACE_TENSION * 2.0 * math.pi),
    }


def run() -> dict[str, Any]:
    _, critical_ratio, critical_alpha = catenoid_critical_parameters()
    branch = []
    for ratio in (0.4, 0.5, 0.6, 0.64, 0.655):
        alpha = catenoid_stable_neck_ratio(ratio)
        converged, neck, area = _solve(ratio, alpha)
        branch.append(
            {
                "D": ratio,
                "converged": converged,
                "neck": neck,
                "exact_neck": alpha,
                "area_ratio": area,
                "exact_area_ratio": catenoid_area_ratio(ratio, alpha),
            }
        )
    return {
        "critical_half_separation_ratio": critical_ratio,
        "critical_neck_ratio": critical_alpha,
        "branch": branch,
        "beyond_limit": _collapse(0.7, critical_alpha),
    }


if __name__ == "__main__":
    print(run())

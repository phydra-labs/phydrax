#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp

from benchmarks._runtime import capture_environment, logical_array_bytes, measure_host
from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)
from phydrax.particle_physics._dark_shower import (
    DarkColorRule,
    DarkShowerEpochPlan,
    DarkSplittingChannel,
    DarkSplittingKernelKind,
    evolve_dark_shower_epoch,
    stage_dark_shower_continuation,
)
from phydrax.particle_physics._events import ParticleEventPlan
from phydrax.particle_physics._identity import ParticleCatalogueReference, ParticleRole
from phydrax.particle_physics._species import ParticleSpeciesTable
from phydrax.particle_physics._weights import EventWeightSet, WeightVariationKind
from phydrax.solver._dark_sector_epoch_runtime import (
    DarkSectorEpochPlan,
    empty_dark_sector_epoch_state,
)
from phydrax.units import COULOMB


def _case(proposal_capacity: int):
    units = RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1),
        RelativityConvention(metric_signature="mostly_minus"),
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros(3),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(1),
        chart_id="flat",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="one-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros(4),
        convention=units.convention,
        source_id="benchmark-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros(4),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="benchmark-observer",
        orientation_id="future-right-handed",
    )
    particle_capacity = 2 * proposal_capacity + 3
    runtime = DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=proposal_capacity,
        product_capacity=particle_capacity,
        radiation_capacity=1,
        work_capacity=particle_capacity,
        frontier_capacity=particle_capacity,
        packet_width=1,
        event_width=8,
        product_width=4,
        radiation_width=1,
        work_width=8,
        frontier_width=8,
        species_revision_id="b" * 64,
        topology_revision_id="c" * 64,
    )
    catalogue = ParticleCatalogueReference(
        source_id="benchmark-dark",
        provider_release="test",
        checksum="checksum",
        citation_url="https://example.test/benchmark-dark",
    )
    species = ParticleSpeciesTable(
        jnp.asarray((100, 101)),
        jnp.asarray((0.0, 0.0)),
        jnp.asarray((1.0, 0.0)),
        catalogue=catalogue,
        energy_unit=units.energy_unit,
        charge_unit=COULOMB,
    )
    plan = DarkShowerEpochPlan(
        runtime,
        species,
        units,
        frame,
        (
            DarkSplittingChannel(
                100,
                (100, 101),
                kernel_kind=DarkSplittingKernelKind.FERMION_VECTOR,
                color_rule=DarkColorRule.FUNDAMENTAL_EMISSION,
                kernel_coefficient=1.0,
                envelope_coefficient=2.0,
            ),
        ),
        ordering="transverse-momentum",
        model_id="benchmark-dark-u1",
        model_revision_id="benchmark-r1",
        tune_id="benchmark-tune",
        alpha_reference=0.2,
        reference_scale=10.0,
        beta0=1.0,
        infrared_cutoff=1.0,
        maximum_scale=10.0,
        z_bounds=(0.1, 0.9),
        proposal_capacity=proposal_capacity,
        production_evidence_ids=("benchmark-dark-shower-control",),
    )
    event_plan = ParticleEventPlan(
        catalogue=catalogue,
        momentum_unit=units.energy_unit,
        length_unit=units.scale.dimensional_scale.length_unit,
        time_unit=units.scale.dimensional_scale.time_unit,
        event_capacity=1,
        particle_capacity=particle_capacity,
        vertex_capacity=proposal_capacity + 1,
        provider_status_namespace="native-dark",
    ).prepare()
    active = jnp.zeros((1, particle_capacity), dtype=bool).at[0, 0].set(True)
    pdg = jnp.zeros((1, particle_capacity), dtype=jnp.int32).at[0, 0].set(100)
    roles = (
        jnp.zeros((1, particle_capacity), dtype=jnp.int32)
        .at[0, 0]
        .set(int(ParticleRole.OUTGOING))
    )
    momenta = jnp.zeros((1, particle_capacity, 4)).at[0, 0].set((100.0, 0.0, 0.0, 100.0))
    color = jnp.zeros((1, particle_capacity, 2), dtype=jnp.int32).at[0, 0].set((1, 0))
    weights = EventWeightSet(
        jnp.ones((1, 1)),
        names=("nominal",),
        variation_kinds=(WeightVariationKind.NOMINAL,),
        correlation_groups=("nominal",),
    )
    events = event_plan.admit(
        event_ids=jnp.asarray((1,)),
        subevent_ids=jnp.asarray((0,)),
        event_active=jnp.asarray((True,)),
        pdg_ids=pdg,
        roles=roles,
        provider_status=jnp.zeros((1, particle_capacity), dtype=jnp.int32),
        momenta=momenta,
        rest_energies=jnp.zeros((1, particle_capacity)),
        particle_active=active,
        mother_indices=jnp.full((1, particle_capacity, 2), -1),
        production_vertex_indices=jnp.full((1, particle_capacity), -1),
        end_vertex_indices=jnp.full((1, particle_capacity), -1),
        color_flow=color,
        production_vertices=jnp.zeros((1, proposal_capacity + 1, 4)),
        vertex_active=jnp.zeros((1, proposal_capacity + 1), dtype=bool),
        weights=weights,
        source_id="benchmark-hard-event",
    )
    uniforms = jnp.tile(jnp.asarray((0.0, 0.5, 0.5, 0.0)), (1, proposal_capacity, 1))
    result, seconds = measure_host(
        lambda: evolve_dark_shower_epoch(plan, events, uniforms)
    )
    continuation, continuation_seconds = measure_host(
        lambda: stage_dark_shower_continuation(
            plan,
            result,
            empty_dark_sector_epoch_state(runtime, epoch_sequence=0),
        )
    )
    return {
        "axes": {
            "event_count": 1,
            "proposal_capacity": proposal_capacity,
            "particle_capacity": particle_capacity,
        },
        "ids": {
            "plan": plan.plan_id,
            "runtime_plan": runtime.plan_id,
            "frame_realization": plan.frame_realization_id,
        },
        "host_seconds": {
            "shower_epoch": seconds,
            "continuation_admission": continuation_seconds,
        },
        "resources": {
            "event_slots": int(result.events.event_active.shape[0]),
            "accepted_branches": int(jnp.sum(result.accepted)),
            "resident_particles": int(jnp.sum(result.events.particle_active)),
            "frontier_branches": int(jnp.sum(result.frontier)),
            "durable_work": int(jnp.sum(continuation.state.work_mask)),
            "durable_frontier": int(jnp.sum(continuation.state.frontier_mask)),
            "backpressured_events": int(jnp.sum(result.backpressured)),
            "logical_bytes": logical_array_bytes(result),
        },
        "scientific_residuals": {
            "maximum_veto_bound_excess": float(
                jnp.max(result.proposal_kernel - result.proposal_envelope)
            ),
        },
        "successful": bool(jnp.all(result.finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-capacities", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.proposal_capacities):
        raise ValueError("proposal capacities must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [_case(value) for value in arguments.proposal_capacities],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()

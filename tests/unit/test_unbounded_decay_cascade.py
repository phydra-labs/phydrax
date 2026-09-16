#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.relativistic_scattering._decays import TwoBodyDecayPlan
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
from phydrax.particle_physics._decay_cascade import (
    DarkDecayCascadePlan,
    DarkDecayChannel,
    DarkDecaySpeciesOwner,
    evolve_decay_cascade_epoch,
    materialize_decay_frontier_work,
    seed_decay_frontier_from_host,
)
from phydrax.particle_physics._host_events import (
    HostEventRecord,
    HostEventWeight,
    HostParticleRecord,
)
from phydrax.particle_physics._identity import (
    ParticleCatalogueReference,
    ParticleRole,
)
from phydrax.particle_physics._species import ParticleSpeciesTable
from phydrax.particle_physics._weights import WeightVariationKind
from phydrax.solver._dark_sector_epoch_runtime import (
    admit_dark_sector_work,
    DarkSectorEpochPlan,
    decode_content_id,
    empty_dark_sector_epoch_state,
)
from phydrax.units import COULOMB


def _contracts(*, product_capacity=2, lifetime=1.0, prompt_cutoff=0.0):
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
        snapshot_token=jnp.asarray(9),
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
        source_id="observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros(4),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="observer",
        orientation_id="future-right-handed",
    )
    runtime = DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=1,
        product_capacity=product_capacity,
        radiation_capacity=1,
        work_capacity=1,
        frontier_capacity=1,
        packet_width=1,
        event_width=8,
        product_width=4,
        radiation_width=1,
        work_width=8,
        frontier_width=8,
        species_revision_id="7" * 64,
        topology_revision_id="8" * 64,
    )
    catalogue = ParticleCatalogueReference(
        source_id="cascade-species",
        provider_release="test",
        checksum="checksum",
        citation_url="https://example.test/cascade",
    )
    species = ParticleSpeciesTable(
        jnp.asarray((30, 32)),
        jnp.asarray((5.0, 0.0)),
        jnp.asarray((0.0, 0.0)),
        catalogue=catalogue,
        energy_unit=units.energy_unit,
        charge_unit=COULOMB,
    )
    channel = DarkDecayChannel(
        TwoBodyDecayPlan(30, (30, 32), (5.0, 0.0), branching_fraction=1.0)
    )
    owner = DarkDecaySpeciesOwner(
        30,
        (channel,),
        owner_id="native-dark-decay",
        mean_proper_lifetime=lifetime,
    )
    cascade = DarkDecayCascadePlan(
        runtime,
        species,
        units,
        frame,
        (owner,),
        model_id="recursive-dark-decay",
        model_revision_id=canonical_fingerprint({"model": "recursive-dark-decay-r1"}),
        prompt_lifetime_cutoff=prompt_cutoff,
        production_evidence_ids=("cascade-control",),
    )
    return cascade


def _event():
    return HostEventRecord(
        11,
        0,
        (HostParticleRecord(0, 30, ParticleRole.OUTGOING, 1, (5.0, 0.0, 0.0, 0.0), 5.0),),
        (),
        (HostEventWeight("nominal", 1.0, WeightVariationKind.NOMINAL, "nominal"),),
        "native-dark-decay",
        "cascade-seed",
    )


def _seed(plan):
    state = empty_dark_sector_epoch_state(plan.runtime_plan, epoch_sequence=0)
    result = seed_decay_frontier_from_host(
        plan,
        state,
        _event(),
        jnp.asarray((0.0,)),
        momentum_unit_id=plan.units.energy_unit.unit_id,
        frame_id=plan.frame.frame_id,
        frame_realization_id=plan.frame_realization_id,
    )
    assert not bool(result.rolled_back)
    assert int(result.state.work_mask.sum()) == 1
    return result.state


def test_finite_epoch_cascade_restarts_to_arbitrary_depth_with_stable_durable_work_ids():
    plan = _contracts()
    state = _seed(plan)
    depth = 20
    previous_work_id = None
    for epoch in range(depth):
        evidence = evolve_decay_cascade_epoch(
            plan, state, 1.0, jnp.zeros((plan.runtime_plan.work_capacity, 5))
        )
        assert bool(evidence.decayed[0])
        assert not bool(evidence.runtime_result.rolled_back)
        assert bool(evidence.runtime_result.conservation_ok)
        np.testing.assert_allclose(evidence.four_momentum_residual[0], 0.0, atol=1e-7)
        durable = materialize_decay_frontier_work(
            plan, evidence, rights_id="test-rights", partition_key="cascade-shard"
        )
        assert len(durable.work_items) == 1
        work_id = durable.work_items[0].work_id
        assert work_id != previous_work_id
        previous_work_id = work_id
        resident = durable.evidence.runtime_result.state
        encoded = resident.frontier_ids[
            np.flatnonzero(np.asarray(resident.frontier_mask))[0]
        ]
        assert decode_content_id(encoded) == work_id
        if epoch + 1 < depth:
            parent_manifest = canonical_fingerprint({"epoch": epoch})
            restarted = empty_dark_sector_epoch_state(
                plan.runtime_plan,
                epoch_sequence=epoch + 1,
                parent_epoch_manifest_id=parent_manifest,
            )
            active = np.flatnonzero(np.asarray(resident.frontier_mask))
            admission = admit_dark_sector_work(
                restarted,
                (work_id,),
                resident.frontier_values[active],
                work_status=resident.frontier_status[active],
            )
            assert not bool(admission.refused)
            state = admission.state
    assert float(resident.frontier_values[0, 7]) == depth


def test_prompt_and_delayed_proper_time_paths_remain_distinct():
    prompt_plan = _contracts(lifetime=0.01, prompt_cutoff=0.1)
    prompt = evolve_decay_cascade_epoch(
        prompt_plan,
        _seed(prompt_plan),
        0.001,
        jnp.zeros((prompt_plan.runtime_plan.work_capacity, 5)),
    )
    assert bool(prompt.prompt[0])
    assert not bool(prompt.delayed[0])

    delayed_plan = _contracts(lifetime=1.0, prompt_cutoff=0.1)
    delayed = evolve_decay_cascade_epoch(
        delayed_plan,
        _seed(delayed_plan),
        1.0,
        jnp.zeros((delayed_plan.runtime_plan.work_capacity, 5)),
    )
    assert not bool(delayed.prompt[0])
    assert bool(delayed.delayed[0])


def test_product_capacity_backpressure_is_atomic_and_retains_parent_frontier():
    plan = _contracts(product_capacity=1)
    state = _seed(plan)
    evidence = evolve_decay_cascade_epoch(
        plan, state, 1.0, jnp.zeros((plan.runtime_plan.work_capacity, 5))
    )
    result = evidence.runtime_result
    assert bool(result.backpressured)
    assert not bool(result.rolled_back)
    assert not bool(evidence.decayed[0])
    assert bool(evidence.deferred[0])
    assert int(result.state.product_mask.sum()) == 0
    assert int(result.state.frontier_mask.sum()) == 1
    np.testing.assert_allclose(
        result.state.frontier_values[0, :4], state.work_values[0, :4]
    )

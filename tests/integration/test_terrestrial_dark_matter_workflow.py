import hashlib

import jax.numpy as jnp
import jax.random as jr

from phydrax.applications.dark_matter._profiles import LayeredTerrestrialProfile
from phydrax.applications.dark_matter._scattering import (
    BoundedThermalMarkSamplerPlan,
    ElasticScatteringTable,
)
from phydrax.applications.dark_matter._terrestrial import TerrestrialTransportPlan
from phydrax.applications.dark_matter._transport import (
    TransportNumericalStatus,
    TransportOutcome,
)
from phydrax.integration import WeightedSampleBatch
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.stochastic import PoissonClockRealization


def _manifest(name):
    payload = name.encode()
    return ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-Test-Only",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="test-only",
        nondimensionalization={
            "length_m": 1.0,
            "thermal_sigma_cutoff": 6.0,
            "maxwellian_tail_probability_bound": 7.488376948795484e-08,
        },
        uncertainty={"relative": 0.0},
        lineage_ids=(f"synthetic:{name}",),
    )


def _plan(*, maximum_guard_events=16):
    profile = LayeredTerrestrialProfile(
        jnp.asarray((1.0, 2.0)),
        jnp.zeros((2,)),
        jnp.zeros((2, 1)),
        jnp.asarray((100.0, 100.0)),
        ("target",),
        _manifest("synthetic-transparent-body"),
        frame_id="synthetic-transparent-frame",
    )
    scattering = ElasticScatteringTable(
        ("target",),
        jnp.asarray((1.0,)),
        jnp.asarray((0.0, 2.0)),
        jnp.zeros((1, 2)),
        _manifest("synthetic-transparent-cross-section"),
        temperatures_K=jnp.asarray((100.0, 200.0)),
        rate_coefficients_m3_s=jnp.zeros((1, 2, 2)),
        mark_sampler=BoundedThermalMarkSamplerPlan(maximum_proposals=32),
    )
    return TerrestrialTransportPlan(
        profile,
        scattering,
        1.0,
        detector_depth_m=0.5,
        maximum_jump_events=8,
        maximum_guard_events=maximum_guard_events,
    )


def _paths(count):
    states = jnp.broadcast_to(
        jnp.asarray((-3.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
        (count, 6),
    )
    return WeightedSampleBatch(
        states,
        jnp.zeros((count,)),
        sample_axes=0,
        provenance="synthetic-unit-flux",
        independent=True,
    )


def _clocks(plan, count):
    return PoissonClockRealization(
        jr.key(12),
        1,
        support=(0.0, 6.0),
        max_events_per_channel=8,
        sample_shape=(count,),
        process_id=plan.process.process_id,
    )


def test_transparent_terrestrial_workflow_replays_guards_and_crossing_measure():
    plan = _plan()
    paths = _paths(4)
    clocks = _clocks(plan, 4)
    times = jnp.asarray((0.0, 6.0))

    first = plan.simulate(paths, clocks, times)
    replay = plan.simulate(paths, clocks, times)

    assert jnp.all(first.evidence.successful)
    assert jnp.all(first.collisions.successful)
    assert jnp.all(first.outcomes == int(TransportOutcome.TRANSMITTED))
    assert jnp.all(jnp.sum(first.outcome_one_hot, axis=-1) == 1)
    assert first.detector_crossings.diagnostics.crossing_count == 8
    assert jnp.allclose(
        jnp.exp(first.detector_crossings.diagnostics.log_total_flux_weight), 8.0
    )
    assert jnp.array_equal(
        first.solution.deterministic_events.event_indices,
        replay.solution.deterministic_events.event_indices,
    )
    assert jnp.array_equal(
        first.solution.deterministic_events.valid,
        replay.solution.deterministic_events.valid,
    )
    assert jnp.allclose(
        first.solution.deterministic_events.event_times,
        replay.solution.deterministic_events.event_times,
        equal_nan=True,
    )


def test_inactive_terrestrial_paths_are_safe_filled_and_excluded():
    plan = _plan()
    base = _paths(2)
    states = jnp.asarray(base.samples).at[1].set(jnp.nan)
    paths = WeightedSampleBatch(
        states,
        base.log_weights,
        mask=jnp.asarray((True, False)),
        support_valid=jnp.asarray(True),
        sample_axes=0,
        provenance="synthetic-masked-flux",
    )
    result = plan.simulate(paths, _clocks(plan, 2), jnp.asarray((0.0, 6.0)))

    assert result.evidence.successful.tolist() == [True, False]
    assert result.outcomes.tolist() == [
        int(TransportOutcome.TRANSMITTED),
        int(TransportOutcome.UNRESOLVED),
    ]
    assert result.detector_crossings.diagnostics.crossing_count == 2


def test_guard_capacity_failure_is_numerical_not_a_transmission_outcome():
    plan = _plan(maximum_guard_events=3)
    result = plan.simulate(_paths(2), _clocks(plan, 2), jnp.asarray((0.0, 6.0)))

    assert jnp.all(
        result.evidence.numerical_status
        == int(TransportNumericalStatus.GUARD_EVENT_CAPACITY)
    )
    assert jnp.all(result.outcomes == int(TransportOutcome.UNRESOLVED))
    assert jnp.all(result.solution.deterministic_events.capacity_exceeded)
    raw_detector = result.solution.deterministic_events.valid & (
        result.solution.deterministic_events.event_indices == plan.detector_in_event_index
    )
    assert jnp.all(jnp.any(raw_detector, axis=-1))
    assert result.detector_crossings.diagnostics.crossing_count == 0

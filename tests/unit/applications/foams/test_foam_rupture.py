import jax.numpy as jnp
import numpy as np

from phydrax.applications.foams import (
    apply_foam_rupture,
    FoamRupturePlan,
    FoamRuptureStatus,
)
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    MultiRegionSurfaceValidationPolicy,
    seed_double_bubble,
    SurfaceEventPolicy,
)
from phydrax.interfacial_transport import FilmStepStatus, SurfaceFilmEvidence


def _problem() -> tuple[
    MultiRegionSurfaceTopology,
    MultiRegionSurfaceState,
    np.ndarray,
    np.ndarray,
]:
    seed = seed_double_bubble(1.0, 0.8, ring_points=12)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-rupture-test"))
    base = seed.state(topology)
    sheet = np.zeros(
        (topology.vertex_capacity, topology.slot_width, 2), dtype=np.float64
    )
    sheet[np.asarray(topology.slot_active), 0] = 1.0e-9
    sheet[np.asarray(topology.slot_active), 1] = 2.0e-6
    region = np.zeros((topology.region_capacity, 2), dtype=np.float64)
    region[: topology.region_count, 0] = (1.0, 2.0, 0.0)
    region[: topology.region_count, 1] = (3.0, 4.0, 0.0)
    state = MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=sheet,
        region_fields=region,
        sheet_field_names=("film_liquid_volume", "circulation"),
        region_field_names=("gas_amount_mol", "gas_internal_energy_j"),
    )
    finite = np.flatnonzero(np.asarray(topology.region_finite))
    pair_index = int(
        np.flatnonzero(
            np.all(
                np.asarray(topology.region_pairs[: topology.region_pair_count])
                == np.sort(finite)[None, :],
                axis=1,
            )
        )[0]
    )
    pair_slots = np.asarray(topology.vertex_pair_slots) == pair_index
    thickness = np.full(
        (topology.vertex_capacity, topology.slot_width), 1.0e-6, dtype=np.float64
    )
    thickness[pair_slots] = 5.0e-8
    return topology, state, thickness, pair_slots


def _film_evidence(
    thickness: np.ndarray,
    rupture_mask: np.ndarray,
    *,
    accepted: bool = True,
    revision: int = 4,
) -> SurfaceFilmEvidence:
    return SurfaceFilmEvidence(
        liquid_volume_residual_m3=jnp.asarray(0.0),
        boundary_exchange_m3=jnp.asarray(0.0),
        minimum_thickness_m=jnp.asarray(np.min(thickness)),
        rupture_mask=jnp.asarray(rupture_mask),
        energy_change_j=jnp.asarray(-1.0),
        dissipation_guaranteed=jnp.asarray(True),
        positivity_guaranteed=jnp.asarray(accepted),
        conductance_admissible=jnp.asarray(accepted),
        nonlinear_status=jnp.asarray(0, dtype=jnp.int32),
        nonlinear_iterations=jnp.asarray(3, dtype=jnp.int32),
        nonlinear_residual_norm=jnp.asarray(1.0e-12),
        converged=jnp.asarray(accepted),
        finite=jnp.asarray(True),
        geometry_revision=jnp.asarray(revision, dtype=jnp.int32),
    )


def test_rupture_requires_accepted_threshold_and_conserves_liquid_to_rim() -> None:
    topology, state, thickness, pair_slots = _problem()
    plan = FoamRupturePlan(5.0e-8)
    evidence = _film_evidence(thickness, pair_slots)
    source_liquid = jnp.sum(
        jnp.where(topology.slot_active, state.sheet_fields[..., 0], 0.0)
    )

    result = apply_foam_rupture(
        plan,
        topology,
        state,
        thickness,
        FilmStepStatus.ACCEPTED,
        evidence,
        4,
        2.5e-9,
    )

    assert result.successful
    assert int(result.evidence.status) == FoamRuptureStatus.COMMITTED
    assert result.topology.region_count == topology.region_count - 1
    assert result.evidence.unresolved_rim_added > 0.0
    assert abs(float(result.evidence.liquid_conservation_residual)) < 1.0e-20
    target_liquid = jnp.sum(
        jnp.where(
            result.topology.slot_active,
            result.state.sheet_fields[..., 0],
            0.0,
        )
    )
    np.testing.assert_allclose(
        target_liquid + result.unresolved_rim_content,
        source_liquid + 2.5e-9,
        rtol=1.0e-14,
    )
    assert float(result.evidence.removed_circulation) > 0.0
    assert abs(float(result.evidence.gas_amount_residual)) < 1.0e-14
    assert abs(float(result.evidence.gas_energy_residual)) < 1.0e-14
    assert result.evidence.gas_region_lineage
    np.testing.assert_allclose(
        jnp.sum(result.state.region_fields[..., 0]),
        jnp.sum(state.region_fields[..., 0]),
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        jnp.sum(result.state.region_fields[..., 1]),
        jnp.sum(state.region_fields[..., 1]),
        rtol=1.0e-14,
    )
    assert not result.evidence.derivative_available


def test_thin_film_from_rejected_step_never_bursts() -> None:
    topology, state, thickness, pair_slots = _problem()
    evidence = _film_evidence(thickness, pair_slots, accepted=False)

    result = apply_foam_rupture(
        FoamRupturePlan(1.0e-7),
        topology,
        state,
        thickness,
        FilmStepStatus.SOLVE_FAILED,
        evidence,
        4,
        0.0,
    )

    assert int(result.evidence.status) == FoamRuptureStatus.FILM_STEP_REJECTED
    assert not bool(result.evidence.triggered)
    assert result.topology is topology
    assert result.state is state
    assert float(result.unresolved_rim_content) == 0.0


def test_burst_at_exact_threshold_is_deterministic() -> None:
    topology, state, thickness, pair_slots = _problem()
    evidence = _film_evidence(thickness, pair_slots)
    plan = FoamRupturePlan(5.0e-8, minimum_trigger_slots=2)

    first = plan.propose(
        topology,
        state,
        thickness,
        FilmStepStatus.ACCEPTED,
        evidence,
        4,
    )
    second = plan.propose(
        topology,
        state,
        thickness,
        FilmStepStatus.ACCEPTED,
        evidence,
        4,
    )

    assert first
    assert tuple(value.proposal_id for value in first) == tuple(
        value.proposal_id for value in second
    )
    assert first[0].minimum_thickness_m == 5.0e-8


def test_failed_candidate_validation_rolls_back_rim_and_surface() -> None:
    topology, state, thickness, pair_slots = _problem()
    evidence = _film_evidence(thickness, pair_slots)
    restrictive = SurfaceEventPolicy(
        validation=MultiRegionSurfaceValidationPolicy(
            check_self_intersection=True,
            intersection_candidate_capacity=1,
        )
    )

    result = apply_foam_rupture(
        FoamRupturePlan(1.0e-7),
        topology,
        state,
        thickness,
        FilmStepStatus.ACCEPTED,
        evidence,
        4,
        7.0e-9,
        event_policy=restrictive,
    )

    assert int(result.evidence.status) == FoamRuptureStatus.TRANSACTION_ROLLED_BACK
    assert result.topology is topology
    assert result.state is state
    assert float(result.unresolved_rim_content) == 7.0e-9
    assert float(result.evidence.unresolved_rim_added) == 0.0

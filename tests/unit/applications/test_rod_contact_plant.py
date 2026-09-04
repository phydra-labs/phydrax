from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.contact._cone import ContactConeSolverPlan
from phydrax.applications.contact._rod_capsule import (
    prepare_reduced_rod_contact_participant,
    RodCapsuleGeometryPlan,
)
from phydrax.applications.contact._rod_contact_lifecycle import (
    CompositeContactResponse,
    PreparedRodContactSearch,
    RodContactCCDPlan,
    RodContactCCDStatus,
    RodContactSearchFailure,
    RodContactSearchPlan,
)
from phydrax.applications.solid_mechanics._rod_contact_plant import (
    FRICTIONLESS_ROD_CONTACT_CAPABILITY,
    ISOTROPIC_COULOMB_ROD_CONTACT_CAPABILITY,
    PreparedReducedRodContactPlant,
    ReducedRodContactPlantStatus,
)
from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan
from phydrax.applications.solid_mechanics._rod_reduced_basis import (
    RodStrainBasisPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_dynamics import (
    prepare_reduced_rod_dynamics,
)
from phydrax.applications.solid_mechanics._rod_reduced_integrators import (
    integrate_reduced_rod_step,
    ReducedRodIntegrationState,
    ReducedRodSemiImplicitVelocityEuler,
)
from phydrax.applications.solid_mechanics._rod_reduction import (
    prepare_reduced_rod,
    ReducedRodPlan,
    ReducedRodState,
)
from phydrax.discretization.contact._implicit_geometry import PlaneContactGeometry
from phydrax.discretization.contact._surface import (
    CollisionFeatureKind,
    CollisionFeaturePolicy,
)
from phydrax.dynamics import PlantStepContext


def _straight_positions(*, height: float = 0.55):
    return jnp.asarray(
        tuple((float(index), 0.0, height) for index in range(6)),
        dtype=jnp.float32,
    )


def _self_contact_positions():
    return jnp.asarray(
        (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (2.0, 0.0, 1.0),
            (2.0, 1.0, 1.0),
            (1.0, 0.22, 1.0),
            (0.0, 0.22, 1.0),
        ),
        dtype=jnp.float32,
    )


def _rod(positions):
    segment_count = positions.shape[0] - 1
    return prepare_rod(
        RodPlan(
            jnp.stack(
                (
                    jnp.arange(segment_count, dtype=jnp.int32),
                    jnp.arange(1, segment_count + 1, dtype=jnp.int32),
                ),
                axis=-1,
            ),
            positions,
            jnp.broadcast_to(jnp.eye(3, dtype=positions.dtype), (segment_count, 3, 3)),
            jnp.ones((segment_count + 1,), dtype=positions.dtype),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((0.2, 0.2, 0.1), dtype=positions.dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((30.0, 12.0, 12.0), dtype=positions.dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((2.0, 2.0, 2.0), dtype=positions.dtype)),
                (segment_count - 1, 3, 3),
            ),
        )
    )


def _plane(offset: float):
    features = CollisionFeaturePolicy(
        jnp.asarray((10_000,), dtype=jnp.int64),
        jnp.asarray((int(CollisionFeatureKind.ANALYTIC),), dtype=jnp.int32),
        participant_ids=101,
        body_ids=103,
        material_ids=107,
        patch_ids=109,
        static_mask=True,
        provenance_id=f"rod-contact-test-plane:{offset}",
    )
    return PlaneContactGeometry(
        jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.float32),
        offset,
        feature_policy=features,
    )


def _prepared_plant(
    *,
    plane_offset: float = -10.0,
    positions=None,
    velocity=None,
    friction: float = 0.0,
    search_capacity: int = 24,
    ccd: RodContactCCDPlan | None = None,
    solver: ContactConeSolverPlan | None = None,
):
    positions = _straight_positions() if positions is None else positions
    rod = _rod(positions)
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        component_scales=jnp.ones((6,), dtype=positions.dtype),
    )
    reduction = prepare_reduced_rod(rod, ReducedRodPlan(basis))
    dynamics = prepare_reduced_rod_dynamics(reduction)
    geometry = RodCapsuleGeometryPlan(
        jnp.full((rod.plan.segment_count,), 0.1, dtype=positions.dtype),
        participant_id=3,
        body_id=5,
        material_id=7,
        patch_id=11,
    ).prepare(rod)
    participant = prepare_reduced_rod_contact_participant(reduction, geometry)
    search = RodContactSearchPlan(
        capacity=search_capacity,
        plane_capacity=rod.plan.segment_count,
        activation_distance=0.04,
        route="dense",
    ).prepare(geometry, planes=(_plane(plane_offset),))
    policy = ReducedRodSemiImplicitVelocityEuler(
        maximum_step_size=0.2,
        energy_balance_tolerance=1.0e3,
    )
    initial = reduction.initialize_state()
    if velocity is not None:
        initial = ReducedRodState(
            initial.coefficients,
            jnp.asarray(velocity, dtype=positions.dtype),
        )
    plant = PreparedReducedRodContactPlant(
        dynamics,
        policy,
        participant,
        search,
        RodContactCCDPlan() if ccd is None else ccd,
        dynamic_friction=friction,
        static_friction=friction,
        cone_solver=solver,
        initial_reduced_state=initial,
        gap_tolerance=2.0e-5,
        energy_tolerance=5.0e-4,
        conservation_tolerance=5.0e-5,
    )
    return plant


def _reset(plant):
    return plant.reset(jax.random.key(31), plant.bind_parameters()).accepted_state


def _step(plant, source, dt=0.025):
    context = PlantStepContext(
        source.time,
        source.time + jnp.asarray(dt, dtype=source.time.dtype),
        source.step_index,
    )
    return plant.step(context, source, None, plant.bind_parameters())


def _assert_tree_exact(actual, expected):
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        actual_is_key = jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key)
        expected_is_key = jax.dtypes.issubdtype(expected_leaf.dtype, jax.dtypes.prng_key)
        assert actual_is_key == expected_is_key
        if actual_is_key:
            actual_leaf = jax.random.key_data(actual_leaf)
            expected_leaf = jax.random.key_data(expected_leaf)
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_contact_free_step_has_free_integrator_parity():
    plant = _prepared_plant(plane_offset=-10.0)
    source = _reset(plant)
    integration_source = ReducedRodIntegrationState(
        source.payload.reduced_state,
        source.payload.material_state,
        source.time,
        source.step_index,
    )
    free = integrate_reduced_rod_step(
        plant.dynamics,
        plant.policy,
        integration_source,
        jnp.asarray(0.01, dtype=source.time.dtype),
        material_control=plant.material_control,
    )

    result = _step(plant, source, 0.01)

    assert result.successful
    assert result.evidence.swept_ccd.evidence.full_step_safe
    assert not jnp.any(result.evidence.event_search.witnesses.valid)
    np.testing.assert_array_equal(
        result.accepted_state.payload.reduced_state.values,
        free.accepted_state.reduced_state.values,
    )
    np.testing.assert_array_equal(
        result.accepted_state.payload.material_state.stretch_shear_history,
        free.accepted_state.material_state.stretch_shear_history,
    )


def test_plane_impact_is_resolved_over_the_requested_interval():
    velocity = jnp.asarray((0.0, 0.0, -12.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
    plant = _prepared_plant(plane_offset=0.0, velocity=velocity)
    source = _reset(plant)

    result = _step(plant, source, 0.05)

    assert result.successful
    assert result.evidence.swept_ccd.evidence.impact_detected
    assert result.evidence.full_interval_covered
    assert jnp.any(result.evidence.response.impulse[:, 0] > 0.0)
    assert result.evidence.final_minimum_gap >= -plant.gap_tolerance
    assert result.accepted_state.time == pytest.approx(source.time + 0.05)
    assert result.accepted_state.step_index == source.step_index + 1


def test_sustained_plane_contact_retains_manifold_and_nonpenetration():
    velocity = jnp.asarray((0.0, 0.0, -12.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
    plant = _prepared_plant(plane_offset=0.0, velocity=velocity)
    first = _step(plant, _reset(plant), 0.05)
    second = _step(plant, first.accepted_state, 0.01)

    assert first.successful & second.successful
    assert second.evidence.final_minimum_gap >= -plant.gap_tolerance
    assert jnp.any(second.accepted_state.payload.contact_state.occupied)
    continued = (
        first.accepted_state.payload.contact_state.route_keys[:, None]
        == second.accepted_state.payload.contact_state.route_keys[None, :]
    )
    assert jnp.any(
        continued
        & first.accepted_state.payload.contact_state.occupied[:, None]
        & second.accepted_state.payload.contact_state.occupied[None, :]
    )


def test_nonadjacent_self_contact_uses_canonical_manifold_routes():
    plant = _prepared_plant(
        positions=_self_contact_positions(),
        plane_offset=-10.0,
    )
    source = _reset(plant)
    positions = plant.participant.positions(source.payload.reduced_state.coefficients)
    search = plant.search.search(positions)

    assert search.successful
    assert search.evidence.adjacency_filtered_count > 0
    assert jnp.any(search.witnesses.valid)
    active_indices = search.witnesses.vertex_indices[search.witnesses.valid]
    left_segments = active_indices[:, :2]
    right_segments = active_indices[:, 2:]
    assert jnp.all(left_segments[:, :, None] != right_segments[:, None, :])
    assert jnp.unique(
        search.witnesses.route_keys[search.witnesses.valid]
    ).size == jnp.sum(search.witnesses.valid)


def test_isotropic_coulomb_response_is_dissipative():
    velocity = jnp.asarray((0.0, 5.0, -12.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
    plant = _prepared_plant(
        plane_offset=0.0,
        velocity=velocity,
        friction=0.6,
    )
    result = _step(plant, _reset(plant), 0.05)

    assert plant.capability_id == ISOTROPIC_COULOMB_ROD_CONTACT_CAPABILITY
    assert result.successful
    assert result.evidence.energy.friction_dissipative
    assert result.evidence.energy.friction_dissipation >= -plant.energy_tolerance
    assert (
        result.evidence.energy.final_mechanical_energy
        <= result.evidence.energy.free_mechanical_energy
        + plant.energy_tolerance * result.evidence.energy.scale
    )
    assert jnp.all(
        jnp.linalg.norm(result.evidence.response.impulse[:, 1:], axis=-1)
        <= 0.6 * jnp.maximum(result.evidence.response.impulse[:, 0], 0.0) + 1.0e-5
    )
    committed_velocity = (
        result.accepted_state.payload.reduced_state.coefficient_velocities
    )
    np.testing.assert_array_equal(
        committed_velocity,
        result.evidence.response.post_velocities[0],
    )
    np.testing.assert_allclose(
        committed_velocity,
        result.evidence.free_step.candidate_state.reduced_state.coefficient_velocities
        + result.evidence.response.velocity_updates[0],
        rtol=8.0 * np.finfo(np.float32).eps,
        atol=8.0 * np.finfo(np.float32).eps,
    )
    response = result.evidence.response
    history = result.accepted_state.payload.contact_state
    active_response = np.flatnonzero(
        np.linalg.norm(np.asarray(response.impulse), axis=-1) > 1.0e-8
    )
    assert active_response.size > 0
    response_interval = (1.0 - float(result.evidence.swept_ccd.impact_fraction)) * 0.05
    for response_index in active_response.tolist():
        matches = np.flatnonzero(
            np.asarray(history.occupied)
            & (np.asarray(history.route_keys) == int(response.route_keys[response_index]))
        )
        assert matches.size == 1
        np.testing.assert_allclose(
            np.asarray(history.slip[int(matches[0])]),
            response_interval * np.asarray(response.slip_velocity[response_index]),
            rtol=2.0e-5,
            atol=2.0e-6,
        )


def _failed_search(original, failure):
    def evaluate(search, positions, /, *, end_positions=None):
        result = original(search, positions, end_positions=end_positions)
        evidence = eqx.tree_at(
            lambda value: (value.complete, value.successful, value.failure),
            result.evidence,
            (
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(int(failure), dtype=jnp.int32),
            ),
        )
        return eqx.tree_at(lambda value: value.evidence, result, evidence)

    return evaluate


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (
            RodContactSearchFailure.CAPACITY_OVERFLOW,
            ReducedRodContactPlantStatus.SEARCH_CAPACITY_EXCEEDED,
        ),
        (
            RodContactSearchFailure.WITNESS_FAILURE,
            ReducedRodContactPlantStatus.SEARCH_FAILED,
        ),
    ),
)
def test_search_and_capacity_failure_roll_back_every_atom(monkeypatch, failure, expected):
    plant = _prepared_plant()
    source = _reset(plant)
    original = PreparedRodContactSearch.search
    monkeypatch.setattr(
        PreparedRodContactSearch,
        "search",
        _failed_search(original, failure),
    )

    result = _step(plant, source)

    assert not result.successful
    assert result.status == int(expected)
    _assert_tree_exact(result.accepted_state.payload, source.payload)
    np.testing.assert_array_equal(result.accepted_state.time, source.time)
    np.testing.assert_array_equal(result.accepted_state.step_index, source.step_index)
    np.testing.assert_array_equal(
        jax.random.key_data(result.accepted_state.key),
        jax.random.key_data(source.key),
    )


def test_certified_safe_prefix_is_never_silently_committed(monkeypatch):
    plant = _prepared_plant()
    source = _reset(plant)
    original = RodContactCCDPlan.evaluate

    def prefix(plan, search, start, end, /, **kwargs):
        result = original(plan, search, start, end, **kwargs)
        evidence = eqx.tree_at(
            lambda value: (
                value.full_step_safe,
                value.impact_detected,
                value.certified_safe_prefix,
                value.status,
            ),
            result.evidence,
            (
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(
                    int(RodContactCCDStatus.CERTIFIED_SAFE_PREFIX), dtype=jnp.int32
                ),
            ),
        )
        result = eqx.tree_at(lambda value: value.evidence, result, evidence)
        return eqx.tree_at(
            lambda value: value.safe_step_fraction,
            result,
            jnp.asarray(0.25, dtype=result.safe_step_fraction.dtype),
        )

    monkeypatch.setattr(RodContactCCDPlan, "evaluate", prefix)
    result = _step(plant, source)

    assert not result.successful
    assert result.status == int(ReducedRodContactPlantStatus.CCD_SAFE_PREFIX_ONLY)
    assert result.candidate_state.time == pytest.approx(source.time + 0.025)
    _assert_tree_exact(result.accepted_state, source)


def test_cone_failure_retains_candidate_iterate_and_rolls_back(monkeypatch):
    velocity = jnp.asarray((0.0, 4.0, -12.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
    plant = _prepared_plant(plane_offset=0.0, velocity=velocity, friction=0.5)
    source = _reset(plant)
    original = CompositeContactResponse.solve

    def fail(response, /, *, initial_impulse=None):
        result = original(response, initial_impulse=initial_impulse)
        evidence = eqx.tree_at(
            lambda value: (value.applied, value.fail_closed, value.successful),
            result.evidence,
            (jnp.asarray(False), jnp.asarray(True), jnp.asarray(False)),
        )
        return eqx.tree_at(lambda value: value.evidence, result, evidence)

    monkeypatch.setattr(CompositeContactResponse, "solve", fail)
    result = _step(plant, source, 0.05)

    assert not result.successful
    assert result.status == int(ReducedRodContactPlantStatus.RESPONSE_SOLVE_FAILED)
    assert result.evidence.response.candidate_impulse.shape == (
        plant.search.plan.total_capacity,
        3,
    )
    _assert_tree_exact(result.accepted_state, source)


def test_checkpoint_replay_reproduces_contact_history_clock_and_key():
    plant = _prepared_plant()
    source = _reset(plant)
    checkpoint = plant.checkpoint(source)
    context = PlantStepContext(
        source.time,
        source.time + jnp.asarray(0.01, dtype=source.time.dtype),
        source.step_index,
    )
    direct = plant.step(context, source, None, plant.bind_parameters())
    digest = plant.state_digest(direct.accepted_state)

    replay = plant.replay(
        checkpoint,
        (context,),
        (None,),
        plant.bind_parameters(),
        expected_digests=(digest,),
    )

    assert replay.matched
    assert replay.first_mismatch_step == -1
    _assert_tree_exact(replay.final_state, direct.accepted_state)


def test_frictionless_capability_id_is_explicit_and_not_a_fallback():
    plant = _prepared_plant(friction=0.0)

    assert plant.capability_id == FRICTIONLESS_ROD_CONTACT_CAPABILITY
    assert jnp.all(plant.dynamic_friction == 0.0)
    assert jnp.all(plant.static_friction == 0.0)
    with pytest.raises(ValueError, match="does not match"):
        PreparedReducedRodContactPlant(
            plant.dynamics,
            plant.policy,
            plant.participant,
            plant.search,
            plant.ccd,
            dynamic_friction=0.0,
            capability_id=ISOTROPIC_COULOMB_ROD_CONTACT_CAPABILITY,
        )

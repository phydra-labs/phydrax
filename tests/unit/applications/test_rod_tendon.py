from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_dynamics import (
    prepare_rod,
    RodPlan,
    RodState,
)
from phydrax.applications.solid_mechanics._rod_reduced_basis import (
    RodStrainBasisPlan,
)
from phydrax.applications.solid_mechanics._rod_reduction import (
    prepare_reduced_rod,
    ReducedRodPlan,
    ReducedRodState,
)
from phydrax.applications.solid_mechanics._rod_tendon import (
    evaluate_tendon_actuation,
    FrictionlessElasticTendonPlan,
    integrate_tendon_payout,
    prepare_frictionless_elastic_tendon,
    prepare_tendon_route,
    RodMaterialStation,
    TendonActuatorState,
    TendonPayoutCommand,
    TendonRoutePlan,
)


def _f32(value):
    return jnp.asarray(value, dtype=jnp.float32)


def _planar_rod(*, stiffness_scale: float = 1.0):
    dtype = jnp.float32
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), dtype=dtype),
            jnp.broadcast_to(jnp.eye(2, dtype=dtype), (2, 2, 2)),
            jnp.asarray((1.0, 1.2, 0.9), dtype=dtype),
            jnp.asarray((0.2, 0.3), dtype=dtype),
            stiffness_scale
            * jnp.broadcast_to(
                jnp.diag(jnp.asarray((80.0, 40.0), dtype=dtype)), (2, 2, 2)
            ),
            stiffness_scale * jnp.asarray((((6.0,),),), dtype=dtype),
        )
    )


def _spatial_rod():
    dtype = jnp.float32
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                dtype=dtype,
            ),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.asarray((1.0, 1.2, 0.9), dtype=dtype),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((80.0, 50.0, 40.0), dtype=dtype)),
                (2, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((7.0, 8.0, 9.0), dtype=dtype)),
                (1, 3, 3),
            ),
        )
    )


def _endpoint_route(offset: float = 0.0, *, label: str | None = None):
    return TendonRoutePlan(
        (
            RodMaterialStation(0, 0.0, jnp.asarray((0.0, offset), dtype=jnp.float32)),
            RodMaterialStation(1, 1.0, jnp.asarray((0.0, offset), dtype=jnp.float32)),
        ),
        label=label,
    )


def _tendon_plan(
    route: TendonRoutePlan,
    *,
    stiffness: float = 10.0,
    maximum_tension: float = 20.0,
    label: str | None = None,
):
    return FrictionlessElasticTendonPlan(
        route,
        stiffness,
        free_length_bounds=(1.0, 3.0),
        payout_rate_bounds=(-0.5, 0.5),
        tendon_length_bounds=(1.5, 2.5),
        maximum_tension=maximum_tension,
        power_tolerance=1.0e-5,
        label=label,
    )


def _planar_reduction(rod):
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=2,
        components=("nu_x", "kappa_z"),
        component_scales=jnp.asarray((0.15, 0.2), dtype=jnp.float32),
    )
    return prepare_reduced_rod(rod, ReducedRodPlan(basis))


def test_material_station_route_preparation_and_content_identities():
    rod = _planar_rod()
    repeated_rod = _planar_rod()
    changed_rod = _planar_rod(stiffness_scale=2.0)
    route = _endpoint_route(0.2, label="display name")
    renamed = _endpoint_route(0.2, label="renamed")
    changed = _endpoint_route(0.25)

    assert route.station_count == 2
    assert route.span_count == 1
    assert route.dimension == 2
    assert route.plan_id == renamed.plan_id
    assert route.plan_id != changed.plan_id
    assert route.stations[0].station_id == renamed.stations[0].station_id
    assert route.stations[0].station_id != changed.stations[0].station_id

    prepared = prepare_tendon_route(route, rod)
    repeated = prepare_tendon_route(renamed, repeated_rod)
    changed_preparation = prepare_tendon_route(route, changed_rod)
    reduction = _planar_reduction(rod)
    reduced = prepare_tendon_route(route, reduction)

    assert prepared.span_count == 1
    assert prepared.reduction is None
    assert prepared.prepared_id == repeated.prepared_id
    assert prepared.workset_id == repeated.workset_id
    assert changed_preparation.prepared_id != prepared.prepared_id
    assert changed_preparation.workset_id == prepared.workset_id
    assert reduced.prepared_id != prepared.prepared_id
    assert reduced.workset_id == prepared.workset_id

    tendon = _tendon_plan(route, label="first")
    renamed_tendon = _tendon_plan(renamed, label="second")
    recalibrated = _tendon_plan(route, stiffness=11.0)
    first_prepared = prepare_frictionless_elastic_tendon(tendon, rod)
    second_prepared = prepare_frictionless_elastic_tendon(renamed_tendon, repeated_rod)

    assert tendon.calibration_id == renamed_tendon.calibration_id
    assert tendon.plan_id == renamed_tendon.plan_id
    assert tendon.plan_id != recalibrated.plan_id
    assert first_prepared.tendon_id == second_prepared.tendon_id
    assert first_prepared.tendon_id != tendon.prepare(reduction).tendon_id
    initialized = first_prepared.initialize_state(_f32(1.8))
    assert initialized.free_length.dtype == jnp.float32
    with pytest.raises(TypeError, match="dtype"):
        first_prepared.initialize_state(jnp.asarray(1.8, dtype=jnp.float16))
    with pytest.raises(TypeError, match="prepared rod dtype"):
        first_prepared.evaluate(
            rod.initialize_state(),
            initialized,
            TendonPayoutCommand(jnp.asarray(0.0, dtype=jnp.float16)),
        )

    with pytest.raises(ValueError, match="outside the rod"):
        prepare_tendon_route(
            TendonRoutePlan(
                (
                    RodMaterialStation(0, 0.0, jnp.zeros((2,), dtype=jnp.float32)),
                    RodMaterialStation(2, 1.0, jnp.zeros((2,), dtype=jnp.float32)),
                )
            ),
            rod,
        )
    with pytest.raises(ValueError, match="dimension"):
        prepare_tendon_route(
            TendonRoutePlan(
                (
                    RodMaterialStation(0, 0.0, jnp.zeros((3,), dtype=jnp.float32)),
                    RodMaterialStation(1, 1.0, jnp.zeros((3,), dtype=jnp.float32)),
                )
            ),
            rod,
        )


def test_exact_material_eyelet_points_velocities_length_and_rate():
    rod = _planar_rod()
    route = prepare_tendon_route(
        TendonRoutePlan(
            (
                RodMaterialStation(0, 0.25, jnp.asarray((0.0, 0.5), dtype=jnp.float32)),
                RodMaterialStation(1, 0.75, jnp.asarray((0.2, -0.1), dtype=jnp.float32)),
            )
        ),
        rod,
    )
    state = RodState(
        jnp.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0)), dtype=jnp.float32),
        jnp.asarray(((1.0, 0.0), (1.0, 2.0), (-1.0, 2.0)), dtype=jnp.float32),
        jnp.asarray((0.5 * jnp.pi, 0.0), dtype=jnp.float32),
        jnp.asarray((2.0, -1.0), dtype=jnp.float32),
    )
    expected_points = jnp.asarray(((0.0, 0.0), (2.2, 1.4)), dtype=jnp.float32)
    expected_velocities = jnp.asarray(((1.0, -0.5), (-0.6, 1.8)), dtype=jnp.float32)
    expected_length = jnp.sqrt(jnp.asarray(6.8, dtype=jnp.float32))
    expected_rate = -0.3 / expected_length

    assert jnp.allclose(route.world_points(state), expected_points, atol=2.0e-6)
    assert jnp.allclose(route.world_velocities(state), expected_velocities, atol=2.0e-6)
    assert route.length(state) == pytest.approx(expected_length, rel=2.0e-6)
    assert route.length_rate(state) == pytest.approx(expected_rate, abs=2.0e-6)
    assert route.native_length_rate_operator(state).mv(
        rod.velocity_from_state(state)
    ) == pytest.approx(expected_rate, abs=2.0e-6)


def test_native_length_jvp_and_true_dual_effort_pullback():
    rod = _planar_rod()
    route = prepare_tendon_route(
        TendonRoutePlan(
            (
                RodMaterialStation(0, 0.2, jnp.asarray((0.1, 0.3), dtype=jnp.float32)),
                RodMaterialStation(1, 0.8, jnp.asarray((-0.2, 0.15), dtype=jnp.float32)),
            )
        ),
        rod,
    )
    state = RodState(
        jnp.asarray(((0.0, 0.0), (1.1, 0.2), (2.0, 0.7)), dtype=jnp.float32),
        jnp.asarray(((0.2, -0.1), (-0.3, 0.4), (0.5, 0.25)), dtype=jnp.float32),
        jnp.asarray((0.25, -0.18), dtype=jnp.float32),
        jnp.asarray((0.35, -0.4), dtype=jnp.float32),
    )

    _, jvp_rate = jax.jvp(
        lambda positions, orientations: route.length(
            RodState(
                positions,
                jnp.zeros_like(positions),
                orientations,
                jnp.zeros_like(orientations),
            )
        ),
        (state.positions, state.orientations),
        (state.velocities, state.angular_velocities),
    )
    length_rate = route.length_rate(state)
    tension = jnp.asarray(3.7, dtype=jnp.float32)
    pullback = route.native_effort_pullback_operator(state)
    effort = pullback.mv(tension)
    rod_power = rod.effort_space.pair(effort, rod.velocity_from_state(state)).real

    assert length_rate == pytest.approx(jvp_rate, rel=2.0e-5, abs=2.0e-6)
    assert pullback.source.space_id == route.tension_space.space_id
    assert pullback.target.space_id == rod.effort_space.space_id
    assert rod_power == pytest.approx(-tension * length_rate, rel=2.0e-5, abs=2.0e-6)
    assert route.tension_space.pair(tension, length_rate) == pytest.approx(
        tension * length_rate
    )


def test_endpoint_and_offset_tendons_have_exact_force_and_bending_signs():
    rod = _planar_rod()
    state = rod.initialize_state()
    tension = jnp.asarray(4.0, dtype=jnp.float32)
    radius = 0.2
    center = prepare_tendon_route(_endpoint_route(0.0), rod)
    upper = prepare_tendon_route(_endpoint_route(radius), rod)
    lower = prepare_tendon_route(_endpoint_route(-radius), rod)

    center_forces, center_moments = center.native_effort(state, tension)
    upper_forces, upper_moments = upper.native_effort(state, tension)
    lower_forces, lower_moments = lower.native_effort(state, tension)
    expected_forces = jnp.asarray(
        ((4.0, 0.0), (0.0, 0.0), (-4.0, 0.0)), dtype=jnp.float32
    )

    assert jnp.allclose(center_forces, expected_forces)
    assert jnp.allclose(upper_forces, expected_forces)
    assert jnp.allclose(lower_forces, expected_forces)
    assert jnp.allclose(center_moments, 0.0)
    assert jnp.allclose(
        upper_moments,
        jnp.asarray((-radius * 4.0, radius * 4.0), dtype=jnp.float32),
    )
    assert jnp.allclose(lower_moments, -upper_moments)
    assert upper_moments[0] < 0.0 < upper_moments[1]
    assert lower_moments[1] < 0.0 < lower_moments[0]

    total_forces = center_forces + upper_forces + lower_forces
    total_moments = center_moments + upper_moments + lower_moments
    assert jnp.allclose(total_forces, 3.0 * expected_forces)
    assert jnp.allclose(total_moments, 0.0)


def test_per_span_rates_and_nonuniform_tensions_preserve_virtual_work():
    rod = _planar_rod()
    route = prepare_tendon_route(
        TendonRoutePlan(
            (
                RodMaterialStation(0, 0.0, jnp.zeros((2,), dtype=jnp.float32)),
                RodMaterialStation(0, 1.0, jnp.zeros((2,), dtype=jnp.float32)),
                RodMaterialStation(1, 1.0, jnp.zeros((2,), dtype=jnp.float32)),
            )
        ),
        rod,
    )
    rest = rod.initialize_state()
    state = RodState(
        rest.positions,
        jnp.asarray(((0.0, 0.0), (0.1, 0.0), (0.4, 0.0)), dtype=jnp.float32),
        rest.orientations,
        rest.angular_velocities,
    )
    tensions = jnp.asarray((2.0, 5.0), dtype=jnp.float32)
    rates = route.span_length_rates(state)
    pullback = route.native_span_effort_pullback_operator(state)
    forces, moments = pullback.mv(tensions)
    power = rod.effort_space.pair((forces, moments), rod.velocity_from_state(state)).real

    assert route.span_count == 2
    assert rates.shape == (2,)
    assert jnp.allclose(rates, jnp.asarray((0.1, 0.3), dtype=jnp.float32))
    assert route.length_rate(state) == pytest.approx(jnp.sum(rates))
    assert pullback.source.space_id == route.span_tension_space.space_id
    assert pullback.target.space_id == rod.effort_space.space_id
    assert jnp.allclose(
        forces,
        jnp.asarray(((2.0, 0.0), (3.0, 0.0), (-5.0, 0.0)), dtype=jnp.float32),
    )
    assert jnp.allclose(moments, 0.0)
    assert power == pytest.approx(-jnp.sum(tensions * rates))

    scalar_effort = route.native_effort(state, jnp.asarray(3.0, dtype=jnp.float32))
    uniform_span_effort = route.native_span_effort(
        state, jnp.full((2,), 3.0, dtype=jnp.float32)
    )
    assert jnp.allclose(scalar_effort[0], uniform_span_effort[0])
    assert jnp.allclose(scalar_effort[1], uniform_span_effort[1])


def test_spatial_offset_transport_and_material_moment_are_power_dual():
    rod = _spatial_rod()
    radius = 0.25
    route = prepare_tendon_route(
        TendonRoutePlan(
            (
                RodMaterialStation(
                    0, 0.0, jnp.asarray((0.0, radius, 0.0), dtype=jnp.float32)
                ),
                RodMaterialStation(
                    1, 1.0, jnp.asarray((0.0, radius, 0.0), dtype=jnp.float32)
                ),
            )
        ),
        rod,
    )
    rest = rod.initialize_state()
    state = RodState(
        rest.positions,
        rest.velocities,
        rest.orientations,
        jnp.asarray(((0.0, 0.0, 2.0), (0.0, 0.0, -1.0)), dtype=jnp.float32),
    )
    tension = jnp.asarray(2.0, dtype=jnp.float32)
    forces, moments = route.native_effort(state, tension)
    point_velocities = route.world_velocities(state)
    effort_power = rod.effort_space.pair(
        (forces, moments), rod.velocity_from_state(state)
    ).real

    assert jnp.allclose(
        route.world_points(state),
        jnp.asarray(((0.0, radius, 0.0), (2.0, radius, 0.0))),
    )
    assert jnp.allclose(
        point_velocities,
        jnp.asarray(((-0.5, 0.0, 0.0), (0.25, 0.0, 0.0))),
    )
    assert route.length_rate(state) == pytest.approx(0.75)
    assert jnp.allclose(
        moments,
        jnp.asarray(((0.0, 0.0, -0.5), (0.0, 0.0, 0.5)), dtype=jnp.float32),
    )
    assert effort_power == pytest.approx(-tension * route.length_rate(state))


def test_slack_taut_continuity_and_all_rating_boundaries():
    rod = _planar_rod()
    route_plan = _endpoint_route()
    prepared = prepare_frictionless_elastic_tendon(_tendon_plan(route_plan), rod)
    rod_state = rod.initialize_state()
    zero_command = TendonPayoutCommand(jnp.asarray(0.0, dtype=jnp.float32))
    epsilon = 1.0e-6

    slack = prepared.evaluate(
        rod_state, TendonActuatorState(_f32(2.0 + epsilon)), zero_command
    )
    boundary = prepared.evaluate(rod_state, TendonActuatorState(_f32(2.0)), zero_command)
    taut = prepared.evaluate(
        rod_state, TendonActuatorState(_f32(2.0 - epsilon)), zero_command
    )
    moving_slack_state = RodState(
        rod_state.positions,
        rod_state.velocities.at[-1, 0].set(0.2),
        rod_state.orientations,
        rod_state.angular_velocities,
    )
    moving_slack = prepared.evaluate(
        moving_slack_state, TendonActuatorState(_f32(2.1)), zero_command
    )

    assert slack.slack and not slack.taut
    assert boundary.slack and not boundary.taut
    assert taut.taut and not taut.slack
    assert slack.tension == pytest.approx(0.0)
    assert boundary.tension == pytest.approx(0.0)
    assert taut.tension == pytest.approx(10.0 * epsilon, abs=2.0e-6)
    assert slack.stored_energy == pytest.approx(0.0)
    assert boundary.stored_energy == pytest.approx(0.0)
    assert taut.stored_energy == pytest.approx(5.0 * epsilon * epsilon, abs=1.0e-10)
    assert moving_slack.length_rate == pytest.approx(0.2)
    assert moving_slack.extension_rate == pytest.approx(0.0)
    assert moving_slack.stored_energy_rate == pytest.approx(0.0)
    assert moving_slack.rod_power == pytest.approx(0.0)

    rated = prepare_frictionless_elastic_tendon(
        _tendon_plan(route_plan, maximum_tension=2.5), rod
    )
    nominal = rated.evaluate(
        rod_state, TendonActuatorState(_f32(1.9)), zero_command, time_step=0.1
    )
    inclusive_rating_boundary = rated.evaluate(
        rod_state,
        TendonActuatorState(_f32(1.75)),
        TendonPayoutCommand(_f32(0.5)),
    )
    inclusive_free_length_boundary = rated.evaluate(
        rod_state, TendonActuatorState(_f32(3.0)), zero_command
    )
    free_length_violation = rated.evaluate(
        rod_state, TendonActuatorState(_f32(0.9)), zero_command
    )
    payout_violation = rated.evaluate(
        rod_state,
        TendonActuatorState(_f32(1.9)),
        TendonPayoutCommand(jnp.asarray(0.6, dtype=jnp.float32)),
    )
    stretched_state = RodState(
        rod_state.positions.at[-1, 0].set(3.0),
        rod_state.velocities,
        rod_state.orientations,
        rod_state.angular_velocities,
    )
    length_violation = rated.evaluate(
        stretched_state, TendonActuatorState(_f32(1.9)), zero_command
    )
    tension_violation = rated.evaluate(
        rod_state, TendonActuatorState(_f32(1.7)), zero_command
    )
    candidate_state_violation = rated.evaluate(
        rod_state,
        TendonActuatorState(_f32(1.9)),
        TendonPayoutCommand(jnp.asarray(0.5, dtype=jnp.float32)),
        time_step=3.0,
    )
    candidate_tension_violation = rated.evaluate(
        rod_state,
        TendonActuatorState(_f32(1.9)),
        TendonPayoutCommand(jnp.asarray(-0.5, dtype=jnp.float32)),
        time_step=1.0,
    )
    invalid_step = rated.evaluate(
        rod_state, TendonActuatorState(_f32(1.9)), zero_command, time_step=-0.1
    )
    degenerate_route = TendonRoutePlan(
        (
            RodMaterialStation(0, 1.0, jnp.zeros((2,), dtype=jnp.float32)),
            RodMaterialStation(1, 0.0, jnp.zeros((2,), dtype=jnp.float32)),
        )
    )
    degenerate = prepare_frictionless_elastic_tendon(
        _tendon_plan(degenerate_route), rod
    ).evaluate(rod_state, TendonActuatorState(_f32(1.0)), zero_command)

    assert nominal.within_rating and nominal.valid
    assert inclusive_rating_boundary.payout_rate_margin == pytest.approx(0.0)
    assert inclusive_rating_boundary.tension_margin == pytest.approx(0.0)
    assert inclusive_rating_boundary.within_rating
    assert inclusive_rating_boundary.valid
    assert inclusive_free_length_boundary.free_length_margin == pytest.approx(0.0)
    assert inclusive_free_length_boundary.state_within_bounds
    assert inclusive_free_length_boundary.valid
    assert not free_length_violation.state_within_bounds
    assert not free_length_violation.within_rating
    assert not payout_violation.payout_rate_within_bounds
    assert not length_violation.tendon_length_within_bounds
    assert not tension_violation.tension_within_bounds
    assert not candidate_state_violation.candidate_state_within_bounds
    assert candidate_state_violation.candidate_state.free_length == pytest.approx(3.4)
    assert not candidate_tension_violation.candidate_tension_within_bounds
    assert not invalid_step.time_step_valid
    assert not invalid_step.valid
    assert not degenerate.nondegenerate
    assert not degenerate.finite
    assert not degenerate.valid


def test_payout_stored_energy_and_rod_spool_power_close_exactly():
    rod = _planar_rod()
    prepared = prepare_frictionless_elastic_tendon(_tendon_plan(_endpoint_route()), rod)
    rest = rod.initialize_state()
    rod_state = RodState(
        rest.positions,
        rest.velocities.at[-1, 0].set(0.2),
        rest.orientations,
        rest.angular_velocities,
    )
    state = TendonActuatorState(jnp.asarray(1.5, dtype=jnp.float32))
    command = TendonPayoutCommand(jnp.asarray(0.05, dtype=jnp.float32))
    evaluation = evaluate_tendon_actuation(
        prepared, rod_state, state, command, time_step=0.1
    )
    integrated = integrate_tendon_payout(prepared, state, command, 0.1)

    assert evaluation.length == pytest.approx(2.0)
    assert evaluation.length_rate == pytest.approx(0.2)
    assert evaluation.extension == pytest.approx(0.5)
    assert evaluation.extension_rate == pytest.approx(0.15)
    assert evaluation.tension == pytest.approx(5.0)
    assert evaluation.stored_energy == pytest.approx(1.25)
    assert evaluation.stored_energy_rate == pytest.approx(0.75)
    assert evaluation.native_rod_power == pytest.approx(-1.0)
    assert evaluation.rod_power == pytest.approx(-1.0)
    assert evaluation.spool_power == pytest.approx(0.25)
    assert evaluation.virtual_work_residual == pytest.approx(0.0, abs=2.0e-6)
    assert evaluation.instantaneous_power_residual == pytest.approx(0.0, abs=2.0e-6)
    assert evaluation.payout_increment == pytest.approx(0.005)
    assert evaluation.candidate_state.free_length == pytest.approx(1.505)
    assert integrated.free_length == pytest.approx(evaluation.candidate_state.free_length)
    assert evaluation.candidate_extension == pytest.approx(0.495)
    assert evaluation.candidate_tension == pytest.approx(4.95)
    assert evaluation.candidate_stored_energy == pytest.approx(1.225125)
    assert evaluation.stored_energy_change == pytest.approx(-0.024875, abs=2.0e-6)
    assert evaluation.spool_work == pytest.approx(0.024875, abs=2.0e-6)
    assert evaluation.discrete_energy_residual == pytest.approx(0.0, abs=2.0e-6)
    assert evaluation.power_balanced
    assert evaluation.valid
    crossing = prepared.evaluate(
        rod_state,
        TendonActuatorState(jnp.asarray(1.9, dtype=jnp.float32)),
        TendonPayoutCommand(jnp.asarray(0.5, dtype=jnp.float32)),
        time_step=1.0,
    )
    assert crossing.taut
    assert crossing.candidate_extension == pytest.approx(0.0)
    assert crossing.candidate_tension == pytest.approx(0.0)
    assert crossing.candidate_stored_energy == pytest.approx(0.0)
    assert crossing.stored_energy_change == pytest.approx(-0.05, abs=2.0e-6)
    assert crossing.spool_work == pytest.approx(0.05, abs=2.0e-6)
    assert crossing.discrete_energy_residual == pytest.approx(0.0, abs=2.0e-6)
    assert crossing.power_balanced
    assert crossing.valid


def test_reduced_length_rate_and_effort_are_exact_pushforward_pullback_duals():
    rod = _planar_rod()
    reduction = _planar_reduction(rod)
    route = prepare_tendon_route(_endpoint_route(0.1), reduction)
    coefficients = jnp.asarray((0.12, -0.18), dtype=jnp.float32)
    rates = jnp.asarray((0.3, -0.4), dtype=jnp.float32)
    state = ReducedRodState(coefficients, rates)
    zero_rates = jnp.zeros_like(rates)

    _, jvp_rate = jax.jvp(
        lambda values: route.length(ReducedRodState(values, zero_rates)),
        (coefficients,),
        (rates,),
    )
    operator = route.reduced_length_rate_operator(state)
    length_rate = operator.mv(rates)
    tension = jnp.asarray(3.2, dtype=jnp.float32)
    pullback = route.reduced_effort_pullback_operator(state)
    reduced_effort = pullback.mv(tension)
    reduced_power = reduction.reduced_effort_space.pair(reduced_effort, rates).real
    native_state = reduction.lift(state)
    native_effort = route.native_effort(native_state, tension)
    expected_effort = reduction.lift_effort_pullback_operator(coefficients).mv(
        native_effort
    )
    native_power = rod.effort_space.pair(
        native_effort, rod.velocity_from_state(native_state)
    ).real

    assert length_rate == pytest.approx(jvp_rate, rel=3.0e-5, abs=3.0e-6)
    assert route.length_rate(state) == pytest.approx(length_rate)
    assert pullback.source.space_id == route.tension_space.space_id
    assert pullback.target.space_id == reduction.reduced_effort_space.space_id
    assert jnp.allclose(reduced_effort, expected_effort, rtol=3.0e-5, atol=3.0e-6)
    assert reduced_power == pytest.approx(-tension * length_rate, rel=3.0e-5, abs=3.0e-6)
    assert native_power == pytest.approx(reduced_power, rel=3.0e-5, abs=3.0e-6)

    geometric_length = route.length(state)
    plan = FrictionlessElasticTendonPlan(
        route.plan,
        10.0,
        free_length_bounds=(0.5, 5.0),
        payout_rate_bounds=(-0.5, 0.5),
        tendon_length_bounds=(0.5, 5.0),
        maximum_tension=100.0,
        power_tolerance=1.0e-5,
    )
    evaluation = plan.prepare(reduction).evaluate(
        state,
        TendonActuatorState(_f32(geometric_length - 0.1)),
        TendonPayoutCommand(jnp.asarray(0.0, dtype=jnp.float32)),
    )

    assert evaluation.reduced_effort is not None
    assert evaluation.reduced_rod_power is not None
    assert jnp.allclose(evaluation.reduced_effort, route.reduced_effort(state, _f32(1.0)))
    assert evaluation.native_rod_power == pytest.approx(
        evaluation.reduced_rod_power, rel=3.0e-5, abs=3.0e-6
    )
    assert evaluation.rod_power == pytest.approx(evaluation.reduced_rod_power)
    assert evaluation.virtual_work_residual == pytest.approx(0.0, abs=3.0e-6)
    assert evaluation.power_balanced


def test_fixed_route_length_and_rate_support_jit_and_vmap():
    rod = _planar_rod()
    route = prepare_tendon_route(_endpoint_route(), rod)
    rest = rod.initialize_state()

    def query(endpoint_y, endpoint_y_velocity):
        state = RodState(
            rest.positions.at[-1, 1].set(endpoint_y),
            rest.velocities.at[-1, 1].set(endpoint_y_velocity),
            rest.orientations,
            rest.angular_velocities,
        )
        return route.length(state), route.length_rate(state)

    endpoint_y = jnp.asarray(0.4, dtype=jnp.float32)
    endpoint_y_velocity = jnp.asarray(0.3, dtype=jnp.float32)
    compiled_length, compiled_rate = jax.jit(query)(endpoint_y, endpoint_y_velocity)
    expected_length = jnp.sqrt(4.0 + endpoint_y * endpoint_y)
    expected_rate = endpoint_y * endpoint_y_velocity / expected_length

    assert compiled_length == pytest.approx(expected_length, rel=2.0e-6)
    assert compiled_rate == pytest.approx(expected_rate, rel=2.0e-6)

    endpoint_ys = jnp.asarray((0.0, 0.3, 0.4), dtype=jnp.float32)
    endpoint_velocities = jnp.asarray((0.2, -0.1, 0.3), dtype=jnp.float32)
    lengths, rates = jax.vmap(query)(endpoint_ys, endpoint_velocities)
    expected_lengths = jnp.sqrt(4.0 + endpoint_ys * endpoint_ys)
    expected_rates = endpoint_ys * endpoint_velocities / expected_lengths

    assert jnp.allclose(lengths, expected_lengths, rtol=2.0e-6, atol=2.0e-6)
    assert jnp.allclose(rates, expected_rates, rtol=2.0e-6, atol=2.0e-6)

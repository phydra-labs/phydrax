from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan
from phydrax.applications.solid_mechanics._rod_floating import (
    FloatingReducedRodDirectLoad,
    FloatingReducedRodPlan,
    FloatingReducedRodPlant,
    FloatingReducedRodPlantControl,
    FloatingReducedRodState,
    prepare_floating_reduced_rod,
)
from phydrax.applications.solid_mechanics._rod_reduced_basis import (
    RodStrainBasisPlan,
)
from phydrax.applications.solid_mechanics._rod_reduction import ReducedRodPlan
from phydrax.dynamics import PlantStepContext


def _rod():
    dtype = jnp.float32
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(
                ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0), (1.7, 0.0, 0.0)),
                dtype=dtype,
            ),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.asarray((0.8, 1.1, 0.9), dtype=dtype),
            jnp.asarray(
                (
                    ((0.20, 0.01, 0.0), (0.01, 0.25, 0.0), (0.0, 0.0, 0.30)),
                    ((0.24, 0.0, 0.01), (0.0, 0.31, 0.0), (0.01, 0.0, 0.27)),
                ),
                dtype=dtype,
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((70.0, 55.0, 45.0), dtype=dtype)),
                (2, 3, 3),
            ),
            jnp.asarray(
                (((8.0, 0.0, 0.0), (0.0, 9.0, 0.0), (0.0, 0.0, 10.0)),),
                dtype=dtype,
            ),
        )
    )


def _reduction_plan(*, label: str | None = None):
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        components=("nu_y", "kappa_z"),
        component_scales=jnp.ones((6,), dtype=jnp.float32),
        label=label,
    )
    return ReducedRodPlan(basis, label=label)


def _prepared(
    *,
    convention: str = "body",
    gravity=None,
    label: str | None = None,
):
    plan = FloatingReducedRodPlan(
        _reduction_plan(label=label), convention=convention, label=label
    )
    return prepare_floating_reduced_rod(_rod(), plan, gravity=gravity)


def _state(
    prepared,
    *,
    base_pose=None,
    coefficients=None,
    base_twist=None,
    rates=None,
):
    initial = prepared.initialize_state()
    return FloatingReducedRodState(
        initial.base_pose if base_pose is None else base_pose,
        initial.coefficients if coefficients is None else coefficients,
        initial.base_twist if base_twist is None else base_twist,
        initial.coefficient_velocities if rates is None else rates,
    )


def _assert_tree_equal(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    assert all(
        jnp.array_equal(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def test_floating_layout_separates_quaternion_point_from_physical_tangent():
    prepared = _prepared()
    state = prepared.initialize_state()
    count = prepared.coordinate_count

    assert state.configuration.shape == (7 + count,)
    assert state.velocity.shape == (6 + count,)
    assert state.values.shape == (13 + 2 * count,)
    assert prepared.configuration_layout.shape == (7 + count,)
    assert prepared.configuration_layout.local_size == 6 + count
    assert prepared.configuration_layout.tangent_size == 6 + count
    assert prepared.state_layout is prepared.configuration_layout
    assert prepared.point_size == 7 + count
    assert prepared.tangent_size == 6 + count
    assert prepared.state_size == 13 + 2 * count
    assert prepared.configuration_geometry.contains(state.configuration)
    assert jnp.array_equal(state.base_pose[:4], jnp.asarray((1.0, 0.0, 0.0, 0.0)))
    assert prepared.plan.convention == "body"
    assert prepared.supports_contact is False


def test_plan_identity_is_content_addressed_and_twist_convention_is_explicit():
    first = FloatingReducedRodPlan(_reduction_plan(label="first"), label="first")
    renamed = FloatingReducedRodPlan(_reduction_plan(label="renamed"), label="renamed")
    spatial = FloatingReducedRodPlan(_reduction_plan(label="first"), convention="spatial")

    assert first.plan_id == renamed.plan_id
    assert spatial.plan_id != first.plan_id
    with pytest.raises(ValueError, match="spatial 3-D"):
        planar_basis = RodStrainBasisPlan.shifted_legendre(
            0,
            dimension=2,
            component_scales=jnp.ones((3,), dtype=jnp.float32),
        )
        FloatingReducedRodPlan(ReducedRodPlan(planar_basis))


def test_free_se3_action_preserves_native_strains_and_sets_the_root_pose():
    prepared = _prepared()
    coefficients = jnp.asarray((0.12, -0.08), dtype=jnp.float32)
    reference = _state(prepared, coefficients=coefficients)
    half = jnp.sqrt(jnp.asarray(0.5, dtype=jnp.float32))
    moved_pose = jnp.asarray((half, 0.0, 0.0, half, 1.3, -0.7, 0.4), dtype=jnp.float32)
    moved = _state(prepared, base_pose=moved_pose, coefficients=coefficients)

    reference_native = prepared.lift(reference)
    moved_native = prepared.lift(moved)
    reference_evaluation = prepared.reduction.rod.evaluate(reference_native)
    moved_evaluation = prepared.reduction.rod.evaluate(moved_native)

    assert jnp.allclose(moved_native.positions[0], moved_pose[4:], atol=2.0e-6)
    assert jnp.allclose(moved_native.orientations[0], moved_pose[:4], atol=2.0e-6)
    assert jnp.allclose(
        moved_evaluation.stretch_shear_strain,
        reference_evaluation.stretch_shear_strain,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert jnp.allclose(
        moved_evaluation.bend_twist_strain,
        reference_evaluation.bend_twist_strain,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_velocity_lift_and_effort_pullback_obey_exact_virtual_power_duality():
    prepared = _prepared()
    state = _state(
        prepared,
        coefficients=jnp.asarray((0.09, -0.05), dtype=jnp.float32),
    )
    tangent = jnp.asarray(
        (0.3, -0.2, 0.1, 0.2, 0.15, -0.1, 0.06, -0.04),
        dtype=jnp.float32,
    )
    native_effort = prepared.reduction.native_effort_space.validate(
        (
            jnp.asarray(
                ((0.4, -0.1, 0.2), (0.2, 0.3, -0.2), (-0.1, 0.2, 0.1)),
                dtype=jnp.float32,
            ),
            jnp.asarray(((0.1, 0.2, -0.1), (-0.2, 0.1, 0.3)), dtype=jnp.float32),
        )
    )
    velocity_operator = prepared.velocity_operator(state)
    pullback = prepared.effort_pullback_operator(state)
    native_velocity = velocity_operator.mv(tangent)
    effort = pullback.mv(native_effort)

    native_power = prepared.reduction.native_effort_space.pair(
        native_effort, native_velocity
    ).real
    floating_power = prepared.effort_space.pair(effort, tangent).real
    assert jnp.allclose(native_power, floating_power, rtol=2.0e-5, atol=2.0e-6)


def test_block_mass_has_base_strain_coupling_and_fixed_base_limit_is_exact():
    prepared = _prepared()
    coefficients = jnp.asarray((0.11, -0.07), dtype=jnp.float32)
    rates = jnp.asarray((0.08, -0.03), dtype=jnp.float32)
    state = _state(prepared, coefficients=coefficients, rates=rates)

    floating_mass = prepared.mass(state)
    fixed_mass = prepared.reduced_dynamics.mass(coefficients)
    floating_bias = prepared.bias(state)
    fixed_bias = prepared.reduced_dynamics.bias(coefficients, rates)
    delegated = prepared.fixed_base_evaluation(state)
    direct = prepared.fixed_base_dynamics.evaluate(state.reduced_state)

    assert floating_mass.matrix.shape == (8, 8)
    assert floating_mass.base_base.shape == (6, 6)
    assert floating_mass.base_reduced.shape == (6, 2)
    assert floating_mass.reduced_base.shape == (2, 6)
    assert floating_mass.reduced_reduced.shape == (2, 2)
    assert jnp.linalg.norm(floating_mass.base_reduced) > 1.0e-5
    assert jnp.allclose(
        floating_mass.reduced_reduced,
        fixed_mass.operator.matrix,
        rtol=4.0e-5,
        atol=4.0e-6,
    )
    assert jnp.allclose(
        floating_bias.effort[6:], fixed_bias.effort, rtol=8.0e-5, atol=8.0e-6
    )
    assert jnp.allclose(
        delegated.forces.total_effort,
        direct.forces.total_effort,
        rtol=2.0e-6,
        atol=2.0e-7,
    )


def test_body_and_spatial_root_twists_use_their_declared_frames():
    body = _prepared(convention="body")
    spatial = _prepared(convention="spatial")
    half = jnp.sqrt(jnp.asarray(0.5, dtype=jnp.float32))
    pose = jnp.asarray((half, 0.0, 0.0, half, 0.0, 0.0, 0.0), dtype=jnp.float32)
    linear_twist = jnp.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
    angular_twist = jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=jnp.float32)

    body_linear = body.lift(_state(body, base_pose=pose, base_twist=linear_twist))
    spatial_linear = spatial.lift(
        _state(spatial, base_pose=pose, base_twist=linear_twist)
    )
    body_angular = body.lift(_state(body, base_pose=pose, base_twist=angular_twist))
    spatial_angular = spatial.lift(
        _state(spatial, base_pose=pose, base_twist=angular_twist)
    )

    assert jnp.allclose(
        body_linear.velocities[0], jnp.asarray((0.0, 1.0, 0.0)), atol=2.0e-6
    )
    assert jnp.allclose(
        spatial_linear.velocities[0], jnp.asarray((1.0, 0.0, 0.0)), atol=2.0e-6
    )
    assert jnp.allclose(
        body_angular.angular_velocities[0],
        jnp.asarray((1.0, 0.0, 0.0)),
        atol=2.0e-6,
    )
    assert jnp.allclose(
        spatial_angular.angular_velocities[0],
        jnp.asarray((0.0, -1.0, 0.0)),
        atol=2.0e-6,
    )


def test_body_convention_block_dynamics_are_se3_objective():
    prepared = _prepared()
    coefficients = jnp.asarray((0.13, -0.09), dtype=jnp.float32)
    twist = jnp.asarray((0.3, -0.1, 0.2, 0.15, -0.2, 0.1), dtype=jnp.float32)
    rates = jnp.asarray((0.07, -0.04), dtype=jnp.float32)
    reference = _state(prepared, coefficients=coefficients, base_twist=twist, rates=rates)
    half = jnp.sqrt(jnp.asarray(0.5, dtype=jnp.float32))
    moved = _state(
        prepared,
        base_pose=jnp.asarray((half, half, 0.0, 0.0, 2.0, -1.0, 0.5), dtype=jnp.float32),
        coefficients=coefficients,
        base_twist=twist,
        rates=rates,
    )

    first = prepared.evaluate(reference)
    second = prepared.evaluate(moved)

    assert jnp.allclose(first.mass.matrix, second.mass.matrix, rtol=8.0e-5, atol=8.0e-6)
    assert jnp.allclose(
        first.energy.kinetic_energy,
        second.energy.kinetic_energy,
        rtol=8.0e-5,
        atol=8.0e-6,
    )
    assert jnp.allclose(
        first.energy.stored_energy,
        second.energy.stored_energy,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert jnp.allclose(first.forces.elastic_effort[:6], 0.0, atol=1.0e-7)
    assert jnp.allclose(second.forces.elastic_effort[:6], 0.0, atol=1.0e-7)


def test_uniform_gravity_is_pure_free_fall_without_strain_acceleration():
    gravity = jnp.asarray((0.0, 0.0, -9.81), dtype=jnp.float32)
    prepared = _prepared(gravity=gravity)
    state = prepared.initialize_state()
    gravity_result = prepared.gravity_effort(state)
    result = prepared.forward_dynamics(state)

    assert result.valid
    assert gravity_result.finite
    assert gravity_result.effort.shape == (prepared.tangent_size,)
    assert jnp.allclose(
        gravity_result.native_effort[0],
        prepared.reduction.rod.node_masses[:, None] * gravity[None, :],
    )
    assert jnp.allclose(result.acceleration[:3], gravity, rtol=2.0e-4, atol=2.0e-4)
    assert jnp.allclose(result.acceleration[3:6], 0.0, atol=3.0e-4)
    assert jnp.allclose(result.acceleration[6:], 0.0, atol=3.0e-4)


def test_force_torque_free_dynamics_preserve_spatial_momentum_instantaneously():
    prepared = _prepared()
    state = _state(
        prepared,
        coefficients=jnp.asarray((0.08, -0.06), dtype=jnp.float32),
        base_twist=jnp.asarray((0.2, -0.15, 0.1, 0.18, -0.12, 0.09), dtype=jnp.float32),
        rates=jnp.asarray((0.05, -0.035), dtype=jnp.float32),
    )
    result = prepared.forward_dynamics(state)
    momentum_rate = prepared.spatial_momentum_rate(state, result.acceleration)
    inverse = prepared.inverse_dynamics(state, result.acceleration)

    assert result.valid
    assert momentum_rate.finite
    assert jnp.allclose(momentum_rate.linear, 0.0, atol=8.0e-4)
    assert jnp.allclose(momentum_rate.angular_about_origin, 0.0, atol=1.5e-3)
    assert jnp.allclose(inverse.required_effort, 0.0, atol=8.0e-4)
    assert jnp.allclose(inverse.residual, 0.0, atol=2.0e-6)


def test_full_direct_effort_enters_the_declared_floating_dual():
    prepared = _prepared()
    state = prepared.initialize_state()
    effort = jnp.linspace(-0.2, 0.3, prepared.tangent_size, dtype=jnp.float32)
    load = FloatingReducedRodDirectLoad(
        effort, source_id="test_effort", power_channel="test_power"
    )
    evaluation = prepared.evaluate(state, direct_loads=(load,))

    assert jnp.allclose(evaluation.forces.direct_effort, effort)
    assert evaluation.forces.source_ids[-1] == "test_effort"
    assert evaluation.forces.channel_names[-1] == "test_power"
    assert evaluation.forces.total_power == pytest.approx(
        jnp.vdot(effort, state.velocity).real
    )


def test_plant_commits_a_complete_step_and_rolls_back_every_atom_on_bad_control():
    prepared = _prepared()
    plant = FloatingReducedRodPlant(prepared)
    parameters = plant.bind_parameters()
    reset = plant.reset(
        jax.random.key(7),
        parameters,
        initial_time=jnp.asarray(0.0, dtype=jnp.float32),
    )
    source = reset.accepted_state
    context = PlantStepContext(
        source.time,
        source.time + jnp.asarray(0.01, dtype=source.time.dtype),
        source.step_index,
    )

    good = plant.step(context, source, plant.zero_control(), parameters)
    assert good.successful
    assert good.accepted_state.step_index == source.step_index + 1
    assert good.accepted_state.time == context.target_time
    assert plant.supports_contact is False

    bad_control = FloatingReducedRodPlantControl(
        jnp.full((prepared.tangent_size,), jnp.nan, dtype=jnp.float32)
    )
    failed = plant.step(context, source, bad_control, parameters)

    assert not failed.successful
    assert failed.status == -2
    assert failed.accepted_state.time == source.time
    assert failed.accepted_state.step_index == source.step_index
    assert jnp.array_equal(failed.accepted_state.key, source.key)
    _assert_tree_equal(failed.accepted_state.payload, source.payload)
    assert plant.state_digest(failed.accepted_state) == plant.state_digest(source)
    assert plant.state_digest(failed.candidate_state) != plant.state_digest(source)

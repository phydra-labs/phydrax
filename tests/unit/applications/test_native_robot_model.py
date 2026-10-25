#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native robot model, scalar joint laws, stops, winding, and URDF lowering."""

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import DimensionalScaleContract
from phydrax.applications.robotics._native_model import (
    lower_native_robot_tree,
    NativeTreeJoint,
    RobotAdaptation,
    RobotCollisionGeometryPlan,
    RobotContactMaterialTable,
    RobotFramePlan,
    RobotNameIDMap,
)
from phydrax.applications.robotics._urdf import (
    parse_urdf_text,
    URDFCollisionPolicy,
    URDFImportError,
    URDFJointMotor,
    URDFStaticPlane,
)
from phydrax.discretization import reduced_forward_dynamics
from phydrax.discretization.contact._articulated_geometry import (
    ArticulatedPrimitiveKind,
)
from phydrax.discretization.particle._rigid_joint_coordinates import (
    LiftedJointCoordinateState,
)
from phydrax.discretization.particle._rigid_joint_laws import (
    ScalarJointDampingPlan,
    ScalarJointMotorPlan,
)
from phydrax.discretization.particle._rigid_unilateral import ReducedJointStopPlan
from phydrax.interchange._report import AdapterStatus


_BODIES = RobotNameIDMap(("base", "carriage", "link"))
_JOINTS = RobotNameIDMap(("elbow", "slide"))
_BASE, _CARRIAGE, _LINK = _BODIES.ids
_ELBOW, _SLIDE = _JOINTS.ids


def _frames() -> RobotFramePlan:
    return RobotFramePlan(
        RobotNameIDMap(("tool",)),
        np.asarray([_LINK], dtype=np.int64),
        np.asarray([[0.5, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )


def _direct_model(**components: Any) -> Any:
    # base (fixed) --slide along x--> carriage --elbow about z at x=1--> link
    return lower_native_robot_tree(
        "direct",
        _BODIES,
        np.asarray([10.0, 2.0, 1.0]),
        np.stack((np.eye(3), 0.2 * np.eye(3), np.diag([0.01, 0.1, 0.1]))),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (3, 1)),
        "base",
        (
            NativeTreeJoint(
                "slide", "prismatic", "base", "carriage", axis=(1.0, 0.0, 0.0)
            ),
            NativeTreeJoint(
                "elbow",
                "hinge",
                "carriage",
                "link",
                anchor=(1.0, 0.0, 0.0),
                axis=(0.0, 0.0, 1.0),
            ),
        ),
        DimensionalScaleContract.si(),
        _frames(),
        **components,
    )


def _dof(prepared: Any, joint_id: int) -> int:
    edges = [int(value) for value in np.asarray(prepared.articulation.joint_ids)]
    return prepared.articulation.joint_velocity_slices[edges.index(joint_id)].start


def test_direct_model_prepares_reference_and_manifest() -> None:
    model = _direct_model()
    prepared = model.prepare()
    articulation = prepared.articulation
    zero = articulation.forward_kinematics(articulation.zero_configuration())

    assert articulation.nq == articulation.nv == 2
    assert zero.finite
    assert jnp.allclose(zero.bodies.position, prepared.reference.position)
    assert model.manifest.capabilities == ("fixed-base-reduced-tree",)
    assert model.manifest.hinge_count == 1
    assert model.manifest.prismatic_count == 1
    assert model.manifest.frame_count == 1
    assert prepared.motors is None and prepared.stops is None
    assert prepared.coordinates.hinge_dof_indices == (_dof(prepared, _ELBOW),)
    assert _direct_model().model_id == model.model_id


def test_named_frame_follows_lifted_hinge_rotation() -> None:
    prepared = _direct_model().prepare()
    articulation = prepared.articulation
    frames = prepared.model.frames
    configuration = jnp.zeros((2,)).at[_dof(prepared, _ELBOW)].set(0.5 * math.pi)
    tool = articulation.frame_transform(
        configuration, int(np.asarray(frames.body_ids)[0]), frames.local_transforms()[0]
    )

    # The link COM sits 0.5 m beyond the elbow and the tool 0.5 m beyond the COM.
    assert jnp.allclose(tool[:3, 3], jnp.asarray([1.0, 1.0, 0.0]), atol=1.0e-12)
    assert jnp.allclose(tool[:3, 0], jnp.asarray([0.0, 1.0, 0.0]), atol=1.0e-12)


def test_compiled_query_receives_the_prepared_articulation_as_dynamic_input() -> None:
    articulation = _direct_model().prepare().articulation
    configuration = jnp.asarray([0.1, 0.4])
    velocity = jnp.asarray([0.2, -0.3])
    effort = jnp.asarray([0.5, -0.1])
    gravity = jnp.asarray([0.0, -9.81, 0.0])

    def query(
        value: Any, q: jax.Array, v: jax.Array, tau: jax.Array, g: jax.Array
    ) -> Any:
        return reduced_forward_dynamics(value, q, v, tau, g)

    # One ahead-of-time executable; the articulation is an argument, not a constant.
    executable = (
        jax.jit(query)
        .lower(articulation, configuration, velocity, effort, gravity)
        .compile()
    )
    result = executable(articulation, configuration, velocity, effort, gravity)
    eager = reduced_forward_dynamics(
        articulation, configuration, velocity, effort, gravity
    )
    assert bool(result.successful) and bool(eager.successful)
    np.testing.assert_allclose(
        result.acceleration, eager.acceleration, rtol=1.0e-10, atol=1.0e-12
    )

    # Updated mass properties reach that same executable: nothing was captured.
    masses = articulation.graph.bodies.mass_properties.masses
    heavier = eqx.tree_at(
        lambda value: value.graph.bodies.mass_properties.masses,
        articulation,
        2.0 * masses,
    )
    updated = executable(heavier, configuration, velocity, effort, gravity)
    expected = reduced_forward_dynamics(heavier, configuration, velocity, effort, gravity)
    assert bool(updated.successful)
    np.testing.assert_allclose(
        updated.acceleration, expected.acceleration, rtol=1.0e-10, atol=1.0e-12
    )
    assert not np.allclose(updated.acceleration, result.acceleration)


def test_massless_world_anchor_is_refused_without_invented_mass() -> None:
    model = _direct_model()
    particles = type(model.particles)(
        np.asarray(_BODIES.ids, dtype=np.int64),
        np.asarray([0.0, 2.0, 1.0]),
        ambient_dimension=3,
        active_mask=np.asarray([False, True, True]),
    )
    with pytest.raises(ValueError, match="massless world anchor"):
        type(model)(
            particles,
            model.bodies,
            model.joints,
            model.articulation,
            model.reference,
            model.dimensions,
            model.body_names,
            model.joint_names,
            model.frames,
        )


def test_motor_transmission_order_saturation_and_power() -> None:
    motors = ScalarJointMotorPlan(
        np.asarray([_ELBOW, _SLIDE, _SLIDE], dtype=np.int64),
        np.asarray([-2.0, 3.0, 1.0]),
        effort_limit=np.asarray([5.0, np.inf, 0.0]),
        command_lower=np.asarray([-1.0, -np.inf, -1.0]),
        command_upper=np.asarray([1.0, np.inf, 1.0]),
    )
    model = _direct_model(
        actuator_names=RobotNameIDMap(("a-elbow", "b-slide", "c-disabled")),
        motors=motors,
    )
    prepared = model.prepare()
    elbow, slide = _dof(prepared, _ELBOW), _dof(prepared, _SLIDE)
    velocity = jnp.zeros((2,)).at[elbow].set(0.5).at[slide].set(-2.0)
    command = jnp.asarray([4.0, 1.5, 0.7])
    assert prepared.motors is not None
    evaluation = prepared.motors.evaluate(command, velocity)

    # Elbow: clip 4 -> 1, gear -2 -> -2, limit 5 keeps -2.
    assert jnp.allclose(evaluation.actuator_effort, jnp.asarray([-2.0, 4.5, 0.0]))
    assert jnp.allclose(evaluation.generalized_effort[elbow], -2.0)
    assert jnp.allclose(evaluation.generalized_effort[slide], 4.5)
    assert evaluation.command_saturated.tolist() == [True, False, False]
    assert evaluation.disabled.tolist() == [False, False, True]
    assert jnp.allclose(evaluation.source_power, -2.0 * 0.5 + 4.5 * -2.0)
    assert jnp.allclose(
        evaluation.source_power, jnp.vdot(evaluation.generalized_effort, velocity)
    )
    compiled = eqx.filter_jit(prepared.motors.evaluate)(command, velocity)
    assert jnp.array_equal(compiled.generalized_effort, evaluation.generalized_effort)
    assert "stateless-scalar-joint-motors" in model.manifest.capabilities


def test_motor_command_gradient_vanishes_only_beyond_saturation() -> None:
    motors = ScalarJointMotorPlan(
        np.asarray([_ELBOW], dtype=np.int64), 2.0, effort_limit=3.0
    )
    prepared = _direct_model(
        actuator_names=RobotNameIDMap(("motor",)), motors=motors
    ).prepare()
    assert prepared.motors is not None
    elbow = _dof(prepared, _ELBOW)

    def effort(command: jax.Array) -> jax.Array:
        assert prepared.motors is not None
        return prepared.motors.evaluate(command, jnp.zeros((2,))).generalized_effort[
            elbow
        ]

    assert jnp.allclose(jax.grad(lambda u: effort(u[None]))(jnp.asarray(1.0)), 2.0)
    assert jnp.allclose(jax.grad(lambda u: effort(u[None]))(jnp.asarray(2.0)), 0.0)


def test_executed_damping_effort_and_dissipation() -> None:
    damping = ScalarJointDampingPlan(
        np.asarray([_ELBOW, _SLIDE], dtype=np.int64), np.asarray([0.3, 4.0])
    )
    prepared = _direct_model(damping=damping).prepare()
    assert prepared.damping is not None
    elbow, slide = _dof(prepared, _ELBOW), _dof(prepared, _SLIDE)
    velocity = jnp.zeros((2,)).at[elbow].set(2.0).at[slide].set(-0.5)
    evaluation = prepared.damping.evaluate(velocity)

    assert jnp.allclose(evaluation.generalized_effort[elbow], -0.6)
    assert jnp.allclose(evaluation.generalized_effort[slide], 2.0)
    assert jnp.allclose(evaluation.dissipation_rate, 0.3 * 4.0 + 4.0 * 0.25)
    assert jnp.allclose(
        jnp.vdot(evaluation.generalized_effort, velocity), -evaluation.dissipation_rate
    )
    assert evaluation.successful


def test_stop_rows_are_affine_dual_and_exact_on_lifted_paths() -> None:
    stops = ReducedJointStopPlan(
        np.asarray([_ELBOW, _SLIDE], dtype=np.int64),
        np.asarray([-np.inf, -0.2]),
        np.asarray([7.0, 0.4]),
    )
    prepared = _direct_model(stops=stops).prepare()
    rows = prepared.stops
    assert rows is not None
    elbow, slide = _dof(prepared, _ELBOW), _dof(prepared, _SLIDE)
    lifted = jnp.zeros((2,)).at[elbow].set(2.0 * math.pi).at[slide].set(0.1)
    gaps = rows.gaps(lifted)

    assert rows.capacity == 4
    assert rows.row_valid.tolist() == [False, True, True, True]
    assert rows.row_units == ("rad", "rad", "m", "m")
    assert jnp.allclose(gaps, jnp.asarray([0.0, 7.0 - 2.0 * math.pi, 0.3, 0.3]))

    velocity = jnp.asarray([0.7, -1.3])
    impulse = jnp.asarray([5.0, 2.0, 3.0, 0.5])
    assert jnp.allclose(
        jnp.vdot(rows.row_velocity(velocity), impulse),
        jnp.vdot(velocity, rows.generalized_impulse(impulse)),
    )
    assert jnp.allclose(rows.generalized_impulse(impulse)[elbow], -2.0)
    operator = rows.jacobian_operator()
    assert jnp.allclose(operator.mv(velocity), rows.row_velocity(velocity))

    start = lifted
    end = lifted.at[slide].set(0.6)
    fractions = jnp.linspace(0.0, 1.0, 33)
    sampled = jax.vmap(lambda s: rows.gaps(start + s * (end - start)))(fractions)
    assert jnp.allclose(
        rows.path_minimum_gaps(start, end), jnp.min(sampled, axis=0), atol=1.0e-14
    )
    assert rows.path_minimum_gaps(start, end)[3] < 0.0


def test_lifted_winding_follows_actual_full_turn_increment() -> None:
    prepared = _direct_model().prepare()
    coordinates = prepared.coordinates
    articulation = prepared.articulation
    elbow = _dof(prepared, _ELBOW)
    source = jnp.zeros((2,)).at[elbow].set(0.25)
    velocity = jnp.zeros((2,)).at[elbow].set(2.0 * math.pi)
    state = coordinates.initial_state()
    update = coordinates.advance(source, state, velocity, jnp.asarray(1.0))

    assert update.successful
    assert jnp.allclose(update.configuration, source, atol=1.0e-12)
    assert jnp.array_equal(
        update.configuration,
        articulation.integrate_configuration(source, velocity, 1.0),
    )
    assert update.accepted_state.winding.tolist()[elbow] == 1
    assert jnp.allclose(
        update.lifted_configuration[elbow], 0.25 + 2.0 * math.pi, atol=1.0e-12
    )
    reverse = coordinates.advance(
        update.configuration, update.accepted_state, -velocity, jnp.asarray(1.0)
    )
    assert reverse.accepted_state.winding.tolist()[elbow] == 0

    halves = coordinates.advance(source, state, velocity, jnp.asarray(0.5))
    continued = coordinates.advance(
        halves.configuration, halves.accepted_state, velocity, jnp.asarray(0.5)
    )
    assert jnp.array_equal(
        continued.accepted_state.winding, update.accepted_state.winding
    )
    principal, split_state, valid = coordinates.split(update.lifted_configuration)
    assert valid
    assert jnp.allclose(principal, update.configuration, atol=1.0e-12)
    assert jnp.array_equal(split_state.winding, update.accepted_state.winding)


def test_lifted_winding_refuses_counter_overflow_and_keeps_history() -> None:
    prepared = _direct_model().prepare(maximum_turns=1)
    coordinates = prepared.coordinates
    elbow = _dof(prepared, _ELBOW)
    state = LiftedJointCoordinateState(jnp.zeros((2,), jnp.int32).at[elbow].set(1))
    velocity = jnp.zeros((2,)).at[elbow].set(2.0 * math.pi)
    update = coordinates.advance(jnp.zeros((2,)), state, velocity, jnp.asarray(1.0))

    assert not update.within_range
    assert not update.successful
    assert jnp.array_equal(update.accepted_state.winding, state.winding)
    _, _, valid = coordinates.split(jnp.zeros((2,)).at[elbow].set(5.0 * math.pi))
    assert not valid


def test_collision_declaration_requires_complete_materials_and_valid_shapes() -> None:
    materials = RobotContactMaterialTable(
        np.asarray([[1, 1]], dtype=np.int64), np.asarray([0.8]), np.asarray([0.6])
    )
    names = RobotNameIDMap(("capsule", "sphere"))
    kinds = (ArticulatedPrimitiveKind.CAPSULE, ArticulatedPrimitiveKind.SPHERE)
    first = np.asarray([[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    second = np.asarray([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    plan = RobotCollisionGeometryPlan(
        names,
        kinds,
        np.asarray([_LINK, _CARRIAGE], dtype=np.int64),
        first,
        second,
        np.asarray([0.05, 0.1]),
        np.asarray([1, 1], dtype=np.int64),
        materials=materials,
        plane_names=RobotNameIDMap(("floor",)),
        plane_normals=np.asarray([[0.0, 0.0, 1.0]]),
        plane_offsets=np.asarray([-1.0]),
        plane_material_ids=np.asarray([1], dtype=np.int64),
    )
    model = _direct_model(collision=plan)

    assert "body-attached-sphere-capsule-collision" in model.manifest.capabilities
    assert "static-plane-collision" in model.manifest.capabilities
    with pytest.raises(ValueError, match="no entry for materials"):
        RobotCollisionGeometryPlan(
            names,
            kinds,
            np.asarray([_LINK, _CARRIAGE], dtype=np.int64),
            first,
            second,
            np.asarray([0.05, 0.1]),
            np.asarray([1, 2], dtype=np.int64),
            materials=materials,
        )
    with pytest.raises(ValueError, match="distinct endpoints"):
        RobotCollisionGeometryPlan(
            names,
            kinds,
            np.asarray([_LINK, _CARRIAGE], dtype=np.int64),
            second,
            second,
            np.asarray([0.05, 0.1]),
            np.asarray([1, 1], dtype=np.int64),
            materials=materials,
        )
    with pytest.raises(ValueError, match="dynamic_friction"):
        RobotContactMaterialTable(
            np.asarray([[1, 1]], dtype=np.int64), np.asarray([0.5]), np.asarray([0.6])
        )


_SPHERE_ARM = """
<robot name="sphere-arm">
  <link name="base">
    <inertial><mass value="4"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <link name="arm">
    <inertial><origin xyz="0.3 0 0"/><mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    <collision name="tip"><origin xyz="0.6 0 0" rpy="0 0.3 0"/>
      <geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <link name="slider">
    <inertial><mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="0" velocity="2"/>
  </joint>
  <joint name="rail" type="prismatic">
    <parent link="arm"/><child link="slider"/><origin xyz="0.6 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.1" upper="0.2" effort="30" velocity="1"/>
    <dynamics damping="1.5"/>
  </joint>
</robot>
"""


_ARM_COLLISION = "/robot/links/arm/collision/0"


def _sphere_policy(**options: Any) -> URDFCollisionPolicy:
    return URDFCollisionPolicy(
        RobotContactMaterialTable(
            np.asarray([[0, 0], [0, 7]], dtype=np.int64),
            np.asarray([0.9, 0.7]),
            np.asarray([0.6, 0.5]),
        ),
        {"arm": 0},
        planes=(URDFStaticPlane("floor", (0.0, 0.0, 1.0), -0.5, 7),),
        **options,
    )


def test_urdf_lowers_sphere_collision_motor_damping_and_explicit_stops() -> None:
    adaptation = parse_urdf_text(
        _SPHERE_ARM,
        root_policy="fixed_world",
        joint_stops="native-unilateral-stops",
        motors=(
            URDFJointMotor("rail-drive", "rail", gear=-2.0, command_upper=10.0),
            URDFJointMotor("shoulder-drive", "shoulder"),
        ),
        collision=_sphere_policy(exclude_jointed_links=True),
    )
    model = adaptation.model
    assert isinstance(adaptation, RobotAdaptation)
    assert adaptation.report.status == AdapterStatus.LOSSLESS
    assert model.collision is not None
    arm = int(model.body_names.id_for_name("arm"))
    assert np.asarray(model.collision.body_ids).tolist() == [arm]
    assert np.allclose(np.asarray(model.collision.local_first), [[0.3, 0.0, 0.0]])
    assert np.allclose(np.asarray(model.collision.radii), [0.05])
    assert adaptation.evidence.collisions[0].source_name == "tip"
    assert adaptation.evidence.collisions[0].origin_rpy_rad == (0.0, 0.3, 0.0)
    assert model.collision.excluded_body_pairs.shape == (2, 2)

    assert model.motors is not None
    assert model.actuator_names.names == ("rail-drive", "shoulder-drive")
    assert np.allclose(np.asarray(model.motors.effort_limit), [30.0, 0.0])
    assert np.allclose(np.asarray(model.motors.gear), [-2.0, 1.0])

    assert model.damping is not None
    rail = int(model.joint_names.id_for_name("rail"))
    assert np.asarray(model.damping.joint_ids).tolist() == [rail]
    assert model.stops is not None
    assert np.allclose(np.asarray(model.stops.lower), [-0.1, -1.0])
    assert np.allclose(np.asarray(model.stops.upper), [0.2, 1.0])

    # Without a collision policy the URDF sphere is an interpretation-changing
    # loss that the caller must waive explicitly.
    evidence_only = parse_urdf_text(
        _SPHERE_ARM,
        root_policy="fixed_world",
        waived_loss_paths=(_ARM_COLLISION,),
    )
    assert evidence_only.report.status == AdapterStatus.DECLARED_LOSS
    assert evidence_only.model.stops is None
    assert evidence_only.model.collision is None
    assert evidence_only.model.motors is None
    assert evidence_only.model.model_id != model.model_id


def test_urdf_requested_collision_refuses_unsupported_shapes_unwaivably() -> None:
    boxed = _SPHERE_ARM.replace(
        '<sphere radius="0.05"/>', '<cylinder radius="0.05" length="0.2"/>'
    )
    path = "/robot/links/arm/collision/0/geometry/cylinder"
    with pytest.raises(URDFImportError) as caught:
        parse_urdf_text(
            boxed,
            root_policy="fixed_world",
            collision=_sphere_policy(),
            waived_loss_paths=(path,),
        )
    assert caught.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert caught.value.report is not None
    assert {loss.path for loss in caught.value.report.negotiation.unwaived_losses} == {
        path
    }

    unrequested = parse_urdf_text(
        boxed,
        root_policy="fixed_world",
        waived_loss_paths=("/robot/links/arm/collision/0",),
    )
    assert unrequested.model.collision is None
    assert unrequested.report.status == AdapterStatus.DECLARED_LOSS


def test_urdf_transmission_is_unwaivable_once_motors_are_requested() -> None:
    transmitted = _SPHERE_ARM.replace("</robot>", '<transmission name="t"/></robot>')
    with pytest.raises(URDFImportError) as caught:
        parse_urdf_text(
            transmitted,
            root_policy="fixed_world",
            motors=(URDFJointMotor("drive", "rail"),),
            collision=_sphere_policy(),
            waived_loss_paths=("/robot/transmission/0",),
        )
    assert caught.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert caught.value.report is not None
    assert caught.value.evidence is not None
    assert caught.value.evidence.motors[0].command_lower == -math.inf
    assert caught.value.report.negotiation.waived_losses == ()
    assert {loss.path for loss in caught.value.report.negotiation.unwaived_losses} == {
        "/robot/transmission/0"
    }
    accepted = parse_urdf_text(
        transmitted,
        root_policy="fixed_world",
        collision=_sphere_policy(),
        waived_loss_paths=("/robot/transmission/0",),
    )
    assert accepted.report.status == AdapterStatus.DECLARED_LOSS
    with pytest.raises(ValueError, match="revolute, continuous, or prismatic"):
        parse_urdf_text(
            _SPHERE_ARM,
            root_policy="fixed_world",
            motors=(URDFJointMotor("drive", "missing"),),
        )


def test_urdf_declaration_order_does_not_change_the_model() -> None:
    reordered = _SPHERE_ARM.replace('<link name="slider">', '<link name="__marker__">')
    # Move the slider link declaration ahead of the base link.
    slider_start = reordered.index('<link name="__marker__">')
    slider_end = reordered.index("</link>", slider_start) + len("</link>")
    slider = reordered[slider_start:slider_end].replace("__marker__", "slider")
    reordered = reordered[:slider_start] + reordered[slider_end:]
    reordered = reordered.replace('<link name="base">', slider + '\n  <link name="base">')
    first = parse_urdf_text(
        _SPHERE_ARM, root_policy="fixed_world", collision=_sphere_policy()
    )
    second = parse_urdf_text(
        reordered, root_policy="fixed_world", collision=_sphere_policy()
    )

    assert first.model.model_id == second.model.model_id
    assert first.evidence.source_id != second.evidence.source_id


def test_robot_adaptation_requires_matching_model_identity() -> None:
    adaptation = parse_urdf_text(
        _SPHERE_ARM, root_policy="fixed_world", collision=_sphere_policy()
    )
    other = _direct_model()
    with pytest.raises(ValueError, match="target the native model"):
        RobotAdaptation(other, adaptation.report, adaptation.evidence)


def test_fixed_descendants_own_no_scalar_law_or_stop_rows() -> None:
    welded = _SPHERE_ARM.replace(
        "</robot>",
        """
  <link name="tool">
    <inertial><origin xyz="0.05 0 0" rpy="0.2 0 0"/><mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
  </link>
  <joint name="weld" type="fixed">
    <parent link="slider"/><child link="tool"/><origin xyz="0.1 0 0" rpy="0 0 0.4"/>
  </joint>
</robot>""",
    )
    model = parse_urdf_text(
        welded, root_policy="fixed_world", collision=_sphere_policy()
    ).model
    prepared = model.prepare()
    weld = model.joint_names.id_for_name("weld")

    assert prepared.articulation.nq == 2
    assert model.manifest.fixed_count == 1
    with pytest.raises(ValueError, match="fixed edge"):
        ReducedJointStopPlan(
            np.asarray([weld], dtype=np.int64), np.asarray([-1.0]), np.asarray([1.0])
        ).prepare(prepared.articulation)
    with pytest.raises(ValueError, match="hinge or prismatic"):
        type(model)(
            model.particles,
            model.bodies,
            model.joints,
            model.articulation,
            model.reference,
            model.dimensions,
            model.body_names,
            model.joint_names,
            model.frames,
            damping=ScalarJointDampingPlan(np.asarray([weld], dtype=np.int64), 1.0),
        )
    with pytest.raises(ValueError, match="revolute, continuous, or prismatic"):
        parse_urdf_text(
            welded,
            root_policy="fixed_world",
            motors=(URDFJointMotor("weld-drive", "weld"),),
        )

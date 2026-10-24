#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compiled-MuJoCo adaptation: admitted mappings and fail-closed refusals.

Expected values are written from the MJCF fixture numbers with independent
axis-angle rotations, not from the adapter's own helpers.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.applications.robotics._mujoco_model import (
    adapt_mujoco_model,
    MujocoContactExclusion,
    MujocoJointStopTransformation,
    MujocoModelAdaptationError,
    MujocoModelEvidence,
    MujocoNativeContactTransformation,
)
from phydrax.applications.robotics._native_model import RobotContactMaterialTable
from phydrax.interchange._report import AdapterStatus


mujoco = pytest.importorskip("mujoco")

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "robotics" / "native_mujoco"


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


def _compile(name: str, /) -> Any:
    return mujoco.MjModel.from_xml_path(str(_FIXTURES / name))


def _axis_angle(axis: tuple[float, float, float], angle: float, /) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    skew = np.asarray(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def _matrix(quaternion: Any, /) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ]
    )


def _materials() -> RobotContactMaterialTable:
    return RobotContactMaterialTable(
        np.asarray([[0, 0]], dtype=np.int64), np.asarray([0.8]), np.asarray([0.6])
    )


def _body_index(adaptation: Any, name: str, /) -> int:
    return int(adaptation.model.body_names.id_for_name(name))


def test_smooth_fixture_maps_compiled_inertials_frames_and_offsets() -> None:
    adaptation = adapt_mujoco_model(_compile("arm_smooth.xml"))
    model = adaptation.model
    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    assert adaptation.report.valid
    assert adaptation.report.status == AdapterStatus.LOSSLESS
    assert adaptation.report.target_id == model.model_id == evidence.target_id
    assert evidence.source_id.startswith("mujoco-mjb-sha256:")
    assert model.body_names.names == ("base", "fore", "tool", "upper")
    assert model.joint_names.names == ("shoulder", "slider", "weld/tool")

    rz = _axis_angle((0.0, 0.0, 1.0), np.pi / 4.0)
    rx = _axis_angle((1.0, 0.0, 0.0), np.pi / 6.0)
    ry = _axis_angle((0.0, 1.0, 0.0), np.pi / 2.0)
    upper_origin = np.asarray([0.0, 0.0, 0.6])
    fore_origin = upper_origin + rz @ np.asarray([0.3, 0.0, 0.0])
    tool_origin = fore_origin + rz @ np.asarray([0.2, 0.0, 0.0])
    expected = {
        "base": (4.0, (0.05, 0.05, 0.05), np.asarray([0.0, 0.0, 0.5]), np.eye(3)),
        "upper": (
            1.5,
            (0.002, 0.012, 0.011),
            upper_origin + rz @ np.asarray([0.15, 0.0, 0.0]),
            rz @ rx,
        ),
        "fore": (
            0.8,
            (0.001, 0.004, 0.0045),
            fore_origin + rz @ np.asarray([0.1, 0.01, 0.0]),
            rz,
        ),
        "tool": (
            0.2,
            (0.0001, 0.0001, 0.00005),
            tool_origin + rz @ ry @ np.asarray([0.0, 0.0, 0.02]),
            rz @ ry,
        ),
    }
    masses = np.asarray(model.particles.masses)
    inertia = np.asarray(model.bodies.inertia_com)
    position = np.asarray(model.reference.position)
    orientation = np.asarray(model.reference.orientation)
    for name, (mass, principal, com, rotation) in expected.items():
        index = _body_index(adaptation, name)
        assert masses[index] == pytest.approx(mass, abs=1e-15)
        np.testing.assert_allclose(inertia[index], np.diag(principal), atol=1e-15)
        np.testing.assert_allclose(position[index], com, atol=1e-12)
        np.testing.assert_allclose(_matrix(orientation[index]), rotation, atol=1e-12)
    assert bool(np.asarray(model.bodies.fixed_mask)[_body_index(adaptation, "base")])

    joints = {joint.name: joint for joint in evidence.joints}
    assert joints["shoulder"].qpos0 == pytest.approx(0.3)
    assert joints["shoulder"].kind == "hinge"
    np.testing.assert_allclose(joints["shoulder"].anchor, upper_origin, atol=1e-12)
    np.testing.assert_allclose(
        joints["shoulder"].axis, rz @ np.asarray([0.0, 1.0, 0.0]), atol=1e-12
    )
    assert joints["slider"].kind == "prismatic"
    np.testing.assert_allclose(joints["slider"].anchor, fore_origin, atol=1e-12)
    np.testing.assert_allclose(
        joints["slider"].axis, rz @ np.asarray([1.0, 0.0, 0.0]), atol=1e-12
    )
    assert joints["weld/tool"].kind == "fixed"
    assert joints["weld/tool"].mujoco_joint_id == -1

    prepared = model.prepare()
    qpos_address, dof_address, qpos0 = evidence.coordinate_map(prepared.articulation)
    by_qpos = dict(zip(qpos_address.tolist(), qpos0.tolist(), strict=True))
    assert by_qpos[joints["shoulder"].qpos_address] == pytest.approx(0.3)
    assert by_qpos[joints["slider"].qpos_address] == 0.0
    assert sorted(dof_address.tolist()) == [0, 1]

    frames = model.frames
    tip = frames.names.names.index("site:tip")
    assert int(np.asarray(frames.body_ids)[tip]) == _body_index(adaptation, "tool")
    np.testing.assert_allclose(
        np.asarray(frames.translation)[tip], [0.0, 0.0, 0.03], atol=1e-15
    )
    upper_frame = frames.names.names.index("body:upper")
    local = np.asarray(frames.local_transforms())[upper_frame]
    np.testing.assert_allclose(local[:3, :3], rx.T, atol=1e-12)
    np.testing.assert_allclose(
        local[:3, 3], rx.T @ np.asarray([-0.15, 0.0, 0.0]), atol=1e-12
    )


def test_smooth_fixture_maps_motor_order_clamps_damping_and_sensors() -> None:
    adaptation = adapt_mujoco_model(_compile("arm_smooth.xml"))
    model = adaptation.model
    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    assert model.actuator_names.names == ("shoulder_motor", "slider_motor")
    motors = model.motors
    assert motors is not None
    shoulder = int(model.joint_names.id_for_name("shoulder"))
    slider = int(model.joint_names.id_for_name("slider"))
    np.testing.assert_array_equal(np.asarray(motors.joint_ids), [shoulder, slider])
    np.testing.assert_allclose(np.asarray(motors.gear), [2.5, -1.5])
    np.testing.assert_allclose(np.asarray(motors.effort_limit), [2.0, np.inf])
    np.testing.assert_allclose(np.asarray(motors.command_lower), [-1.0, -2.0])
    np.testing.assert_allclose(np.asarray(motors.command_upper), [1.0, 3.0])
    damping = model.damping
    assert damping is not None
    coefficients = dict(
        zip(
            np.asarray(damping.joint_ids).tolist(),
            np.asarray(damping.damping).tolist(),
            strict=True,
        )
    )
    assert coefficients == {shoulder: pytest.approx(0.2), slider: pytest.approx(0.5)}
    assert model.stops is None
    assert model.collision is None
    assert evidence.integrator == "euler"
    assert evidence.timestep == pytest.approx(0.002)
    assert "eulerdamp" in evidence.disabled_flags
    assert evidence.contact_mode == "no-compiled-contact"
    sensors = {sensor.name: sensor for sensor in evidence.sensors}
    assert sensors["shoulder_q"].quantity == "joint-position"
    assert sensors["shoulder_q"].units == "rad"
    assert sensors["slider_v"].units == "m/s"
    assert sensors["slider_v"].stage == "velocity"
    assert sensors["tip_pos"].target == "site:tip"
    assert sensors["clock"].stage == "clock"


def test_compiled_identity_is_deterministic_and_model_is_provider_free() -> None:
    first = adapt_mujoco_model(_compile("arm_smooth.xml"))
    second = adapt_mujoco_model(_compile("arm_smooth.xml"))
    assert first.evidence.source_id == second.evidence.source_id
    assert first.model.model_id == second.model.model_id
    assert first.evidence.evidence_id == second.evidence.evidence_id
    for leaf in jax.tree_util.tree_leaves(first.model):
        assert not type(leaf).__module__.startswith("mujoco")


def test_contact_requires_explicit_declared_policy() -> None:
    with pytest.raises(MujocoModelAdaptationError, match="executes MuJoCo contact"):
        adapt_mujoco_model(_compile("arm_contact.xml"))
    excluded = adapt_mujoco_model(
        _compile("arm_contact.xml"),
        contact=MujocoContactExclusion("qualification without contact"),
    )
    assert excluded.model.collision is None
    losses = {loss.path: loss for loss in excluded.report.losses}
    assert losses["/contact"].category == "dropped"
    assert losses["/contact"].changes_interpretation
    assert excluded.report.valid


def test_native_contact_transformation_keeps_geometry_and_pair_filters() -> None:
    adaptation = adapt_mujoco_model(
        _compile("arm_contact.xml"),
        contact=MujocoNativeContactTransformation(
            _materials(), "declared native hard contact", default_material_id=0
        ),
    )
    collision = adaptation.model.collision
    assert collision is not None
    assert collision.names.names == ("foot_ball", "link_capsule")
    capsule = collision.names.names.index("link_capsule")
    ball = collision.names.names.index("foot_ball")
    assert int(collision.kinds[capsule]) == 1
    assert int(collision.kinds[ball]) == 0
    endpoints = {
        tuple(np.round(np.asarray(collision.local_first)[capsule], 12)),
        tuple(np.round(np.asarray(collision.local_second)[capsule], 12)),
    }
    assert endpoints == {(0.0, 0.0, 0.2), (0.0, 0.0, -0.2)}
    np.testing.assert_allclose(
        np.asarray(collision.local_first)[ball], [0.0, 0.0, -0.05], atol=1e-15
    )
    np.testing.assert_allclose(np.asarray(collision.radii), [0.04, 0.03])
    assert collision.plane_names.names == ("floor",)
    np.testing.assert_allclose(np.asarray(collision.plane_normals), [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(np.asarray(collision.plane_offsets), [0.0], atol=1e-15)
    link = _body_index(adaptation, "link")
    foot = _body_index(adaptation, "foot")
    np.testing.assert_array_equal(
        np.asarray(collision.excluded_body_pairs), [sorted((link, foot))]
    )
    loss = {item.path: item for item in adaptation.report.losses}["/contact"]
    assert loss.category == "transformed"
    assert loss.changes_interpretation
    assert adaptation.report.valid
    assert adaptation.evidence.contact_mode == "native-inelastic-coulomb-hard-contact"


def test_limits_require_declared_stop_transformation_with_lifted_bounds() -> None:
    with pytest.raises(MujocoModelAdaptationError, match="joint limits"):
        adapt_mujoco_model(_compile("arm_limited.xml"))
    adaptation = adapt_mujoco_model(
        _compile("arm_limited.xml"),
        joint_stops=MujocoJointStopTransformation("declared hard stops"),
    )
    stops = adaptation.model.stops
    assert stops is not None
    bounds = {
        adaptation.model.joint_names.name_for_id(int(identifier)): (low, high)
        for identifier, low, high in zip(
            np.asarray(stops.joint_ids),
            np.asarray(stops.lower),
            np.asarray(stops.upper),
            strict=True,
        )
    }
    assert bounds["pivot"] == (pytest.approx(-0.75), pytest.approx(0.75))
    assert bounds["rail"] == (pytest.approx(-0.05), pytest.approx(0.1))
    loss = {item.path: item for item in adaptation.report.losses}["/joints/limits"]
    assert loss.category == "transformed"
    assert loss.changes_interpretation


_BASE = """
<mujoco>
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
  {option}
  <worldbody>
    <body name="base" pos="0 0 1">
      <inertial pos="0 0 0" mass="{base_mass}" diaginertia="0.1 0.1 0.1"/>
      <body name="link" pos="0 0 0">
        {joint}
        <inertial pos="0.1 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        {geom}
      </body>
    </body>
    {world}
  </worldbody>
  {extra}
</mujoco>
"""


def _refusal_source(**overrides: str) -> str:
    fields = {
        "option": "",
        "base_mass": "2",
        "joint": '<joint name="hinge" type="hinge" axis="0 1 0"/>',
        "geom": "",
        "world": "",
        "extra": "",
    }
    fields.update(overrides)
    return _BASE.format(**fields)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"joint": '<joint name="ball" type="ball"/>'}, "hinge or slide"),
        (
            {
                "joint": '<joint name="a" type="hinge" axis="0 1 0"/>'
                '<joint name="b" type="hinge" axis="1 0 0"/>'
            },
            "multiple joints",
        ),
        (
            {"joint": '<joint name="hinge" type="hinge" axis="0 1 0" stiffness="3"/>'},
            "passive joint springs",
        ),
        (
            {"joint": '<joint name="hinge" type="hinge" axis="0 1 0" armature="0.1"/>'},
            "armature",
        ),
        (
            {
                "joint": '<joint name="hinge" type="hinge" axis="0 1 0" '
                'frictionloss="0.1"/>'
            },
            "friction loss",
        ),
        ({"base_mass": "0"}, "positive compiled mass"),
        (
            {
                "world": '<body name="other" pos="1 0 0"><inertial pos="0 0 0" '
                'mass="1" diaginertia="1 1 1"/></body>'
            },
            "Exactly one compiled body",
        ),
        ({"option": '<option density="1.2"/>'}, "Fluid"),
        (
            {"extra": '<actuator><position joint="hinge" kp="10"/></actuator>'},
            "bias",
        ),
        (
            {
                "extra": '<actuator><motor joint="hinge" forcerange="-1 2"/>'
                "</actuator>"
            },
            "asymmetric force range",
        ),
        (
            {"extra": '<actuator><velocity joint="hinge" kv="1"/></actuator>'},
            "bias",
        ),
        (
            {
                "extra": '<equality><joint joint1="hinge"/></equality>'
            },
            "equality",
        ),
        (
            {
                "extra": '<sensor><jointpos joint="hinge" noise="0.1"/></sensor>'
            },
            "noise",
        ),
        (
            {
                "extra": '<sensor><accelerometer site="s"/></sensor>',
                "geom": '<site name="s"/>',
            },
            "no native observation",
        ),
    ],
)
def test_unsupported_compiled_features_are_refused(
    overrides: dict[str, str], message: str
) -> None:
    model = mujoco.MjModel.from_xml_string(_refusal_source(**overrides))
    with pytest.raises(MujocoModelAdaptationError, match=message):
        adapt_mujoco_model(model)


def test_colliding_box_is_refused_under_native_contact() -> None:
    model = mujoco.MjModel.from_xml_string(
        _refusal_source(
            geom='<geom name="block" type="box" size="0.05 0.05 0.05"/>',
            world='<geom name="floor" type="plane" size="0 0 1"/>',
        )
    )
    with pytest.raises(MujocoModelAdaptationError, match="no native primitive"):
        adapt_mujoco_model(
            model,
            contact=MujocoNativeContactTransformation(
                _materials(), "declared", default_material_id=0
            ),
        )


def test_damper_disabled_in_source_is_not_executed_natively() -> None:
    model = mujoco.MjModel.from_xml_string(
        _refusal_source(
            option='<option><flag damper="disable"/></option>',
            joint='<joint name="hinge" type="hinge" axis="0 1 0" damping="0.4"/>',
        )
    )
    adaptation = adapt_mujoco_model(model)
    assert adaptation.model.damping is None
    assert not adaptation.evidence.damping_executed


def test_floating_base_is_refused() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="floating">
              <freejoint name="free"/>
              <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    with pytest.raises(MujocoModelAdaptationError, match="floating bases"):
        adapt_mujoco_model(model)


def test_pair_inventory_budget_is_checked_before_scanning() -> None:
    model = _compile("arm_contact.xml")
    policy = MujocoNativeContactTransformation(
        _materials(), "declared", default_material_id=0
    )
    with pytest.raises(MujocoModelAdaptationError, match="maximum_geom_pair_checks"):
        adapt_mujoco_model(model, contact=policy, maximum_geom_pair_checks=2)
    adaptation = adapt_mujoco_model(model, contact=policy, maximum_geom_pair_checks=3)
    assert adaptation.model.collision is not None

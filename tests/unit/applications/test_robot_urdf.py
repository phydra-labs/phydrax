#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.robotics._urdf import (
    parse_urdf_file,
    parse_urdf_text,
    URDFImportError,
)
from phydrax.interchange._report import AdapterStatus


_TWO_LINK = """
<robot name="arm">
  <link name="tool">
    <inertial>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <mass value="2"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/>
    </inertial>
  </link>
  <link name="base">
    <inertial>
      <mass value="5"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1.1" iyz="0" izz="1.2"/>
    </inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="tool"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 2 0"/>
    <limit lower="-1" upper="1.5" effort="20" velocity="3"/>
    <dynamics damping="0.25"/>
  </joint>
</robot>
"""


def _prepare(adaptation):
    particles = adaptation.particles.prepare()
    bodies = adaptation.bodies.prepare(particles)
    reference = bodies.kinematics(
        adaptation.reference.position,
        adaptation.reference.velocity,
        adaptation.reference.orientation,
        adaptation.reference.angular_velocity,
    )
    graph = adaptation.joints.prepare(bodies, reference)
    articulation = adaptation.articulation.prepare(graph, reference)
    return bodies, graph, articulation, reference


def _link(name: str, *, body_mass: float = 1.0, inertia: str = "1 1 1") -> str:
    ixx, iyy, izz = inertia.split()
    return f"""
    <link name="{name}">
      <inertial>
        <mass value="{body_mass}"/>
        <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
      </inertial>
    </link>
    """


def test_two_link_urdf_builds_native_plans_reference_axis_and_zero_fk():
    adaptation = parse_urdf_text(_TWO_LINK)

    assert adaptation.report.status == AdapterStatus.LOSSLESS
    assert adaptation.report.valid
    assert adaptation.negotiation.valid
    assert adaptation.dimensions.length_unit == "m"
    assert adaptation.dimensions.mass_unit == "kg"
    assert adaptation.dimensions.time_unit == "s"
    assert adaptation.link_ids.names == ("base", "tool")
    assert adaptation.joint_ids is not None
    assert adaptation.joint_ids.names == ("shoulder",)
    assert np.asarray(adaptation.link_ids.ids).dtype == np.dtype(np.int64)
    assert np.asarray(adaptation.joint_ids.ids).dtype == np.dtype(np.int64)
    assert np.allclose(np.asarray(adaptation.particles.masses), [5.0, 2.0])
    assert np.allclose(
        np.asarray(adaptation.bodies.inertia_body),
        np.asarray((np.diag([1.0, 1.1, 1.2]), np.diag([0.2, 0.3, 0.4]))),
    )
    assert adaptation.joints.hinge is not None
    assert np.allclose(np.asarray(adaptation.joints.hinge.reference_axes), [[0.0, 1.0, 0.0]])
    assert np.allclose(np.asarray(adaptation.joints.hinge.reference_anchors), [[0.0, 0.0, 1.0]])
    assert np.allclose(
        np.asarray(adaptation.reference.position), [[0.0, 0.0, 0.0], [0.1, 0.0, 1.0]]
    )
    joint_evidence = adaptation.evidence.joints[0]
    assert joint_evidence.lower_limit == -1.0
    assert joint_evidence.upper_limit == 1.5
    assert joint_evidence.effort_limit == 20.0
    assert joint_evidence.velocity_limit == 3.0
    assert joint_evidence.damping == 0.25

    _, _, articulation, reference = _prepare(adaptation)
    assert articulation.nq == articulation.nv == 1
    zero = articulation.forward_kinematics(articulation.zero_configuration())
    assert zero.finite
    assert jnp.allclose(zero.bodies.position, reference.position)
    assert jnp.allclose(zero.bodies.orientation, reference.orientation)


def test_com_recentering_and_inertial_frame_rotation_are_exact():
    text = f"""
    <robot name="frames">
      {_link("base")}
      <link name="child">
        <inertial>
          <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
          <mass value="2"/>
          <inertia ixx="1" ixy="0" ixz="0" iyy="2" iyz="0" izz="2.5"/>
        </inertial>
      </link>
      <joint name="mount" type="fixed">
        <parent link="base"/>
        <child link="child"/>
        <origin xyz="2 0 0" rpy="0 0 1.5707963267948966"/>
      </joint>
    </robot>
    """
    adaptation = parse_urdf_text(text)
    child_id = int(adaptation.link_ids.id_for_name("child"))

    assert np.allclose(
        np.asarray(adaptation.reference.position)[child_id], [2.0, 1.0, 0.0], atol=1.0e-14
    )
    assert np.allclose(
        np.asarray(adaptation.reference.orientation)[child_id],
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1.0e-14,
    )
    assert np.allclose(
        np.asarray(adaptation.bodies.inertia_body)[child_id],
        np.diag([2.0, 1.0, 2.5]),
        atol=1.0e-14,
    )
    child = next(item for item in adaptation.evidence.links if item.name == "child")
    assert child.com_in_link_frame_m == (1.0, 0.0, 0.0)
    assert child.link_frame_in_body_m == (-1.0, -0.0, -0.0)


def test_name_maps_target_and_report_are_deterministic():
    first = parse_urdf_text(_TWO_LINK)
    second = parse_urdf_text(_TWO_LINK)

    assert first.link_name_to_id == second.link_name_to_id
    assert first.joint_name_to_id == second.joint_name_to_id
    assert first.target_id == second.target_id
    assert first.evidence.evidence_id == second.evidence.evidence_id
    assert first.report.report_id == second.report.report_id
    assert first.report.negotiation.negotiation_id == second.report.negotiation.negotiation_id


def test_visual_is_declared_optional_loss_but_collision_requires_explicit_waiver():
    visual = _TWO_LINK.replace(
        "<inertial>\n      <mass value=\"5\"/>",
        "<visual><geometry><box size=\"1 1 1\"/></geometry></visual>\n    <inertial>\n      <mass value=\"5\"/>",
    )
    optional = parse_urdf_text(visual)
    assert optional.report.status == AdapterStatus.DECLARED_LOSS
    assert optional.report.valid
    assert len(optional.report.losses) == 1
    assert optional.report.losses[0].path == "/robot/links/base/visual/0"
    assert not optional.report.losses[0].changes_interpretation

    collision = visual.replace("visual", "collision")
    with pytest.raises(URDFImportError) as caught:
        parse_urdf_text(collision)
    assert caught.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert caught.value.report is not None
    assert not caught.value.report.valid
    assert caught.value.evidence is not None

    waived = parse_urdf_text(
        collision, waived_loss_paths=("/robot/links/base/collision/0",)
    )
    assert waived.report.status == AdapterStatus.DECLARED_LOSS
    assert waived.report.valid
    assert len(waived.report.negotiation.waived_losses) == 1
    assert waived.report.negotiation.waived_losses[0].path == (
        "/robot/links/base/collision/0"
    )
    assert len(waived.report.waivers) == 1


@pytest.mark.parametrize(
    "unsafe",
    (
        '<!DOCTYPE robot [<!ENTITY secret SYSTEM "file:///etc/passwd">]><robot name="x">&secret;</robot>',
        '<?xml-stylesheet href="http://example.test/a.xsl"?><robot name="x"/>',
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="x">'
        + _link("base")
        + '<xacro:include filename="other.xacro"/></robot>',
    ),
)
def test_declarations_entities_processing_and_unwaived_extensions_fail_closed(unsafe):
    with pytest.raises(URDFImportError) as caught:
        parse_urdf_text(unsafe)
    assert caught.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC


def test_file_resolver_normalizes_within_root_and_rejects_traversal(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "arm.urdf"
    source.write_text(_TWO_LINK, encoding="utf-8")
    outside = tmp_path / "outside.urdf"
    outside.write_text(_TWO_LINK, encoding="utf-8")

    loaded = parse_urdf_file("./arm.urdf", allowed_root=allowed)
    assert loaded.evidence.source_path == str(source.resolve())

    with pytest.raises(URDFImportError) as caught:
        parse_urdf_file("../outside.urdf", allowed_root=allowed)
    assert caught.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC

    with pytest.raises(URDFImportError) as network:
        parse_urdf_file("https://example.test/arm.urdf", allowed_root=allowed)
    assert network.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC


def test_size_limit_is_enforced_for_text_and_files(tmp_path: Path):
    with pytest.raises(URDFImportError) as text_error:
        parse_urdf_text(_TWO_LINK, max_bytes=16)
    assert text_error.value.status == AdapterStatus.MALFORMED_SOURCE

    source = tmp_path / "arm.urdf"
    source.write_text(_TWO_LINK, encoding="utf-8")
    with pytest.raises(URDFImportError) as file_error:
        parse_urdf_file(source, allowed_root=tmp_path, max_bytes=16)
    assert file_error.value.status == AdapterStatus.MALFORMED_SOURCE


@pytest.mark.parametrize(
    "joint_type",
    ("floating", "planar"),
)
def test_unsupported_joint_kinds_reject(joint_type):
    text = f"""
    <robot name="unsupported">
      {_link("base")}
      {_link("child")}
      <joint name="j" type="{joint_type}">
        <parent link="base"/><child link="child"/>
      </joint>
    </robot>
    """
    with pytest.raises(URDFImportError) as caught:
        parse_urdf_text(text)
    assert caught.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC


@pytest.mark.parametrize(
    "body",
    (
        (
            '<link name="base"><inertial><mass value="1"/>'
            '<inertia ixx="-1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>'
            "</inertial></link>"
        ),
        (
            '<link name="base"><inertial><mass value="1"/>'
            '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="3"/>'
            "</inertial></link>"
        ),
        (
            '<link name="base"><inertial><mass value="0"/>'
            '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>'
            "</inertial></link>"
        ),
    ),
)
def test_invalid_mass_and_inertia_reject_as_malformed(body):
    with pytest.raises(URDFImportError) as caught:
        parse_urdf_text(f'<robot name="bad">{body}</robot>')
    assert caught.value.status == AdapterStatus.MALFORMED_SOURCE


@pytest.mark.parametrize(
    "topology",
    (
        """
        <joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint>
        <joint name="ba" type="fixed"><parent link="b"/><child link="a"/></joint>
        """,
        """
        <joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint>
        """,
    ),
)
def test_cyclic_and_disconnected_models_reject(topology):
    links = _link("a") + _link("b") + _link("c")
    with pytest.raises(URDFImportError) as caught:
        parse_urdf_text(f'<robot name="bad-tree">{links}{topology}</robot>')
    assert caught.value.status == AdapterStatus.INCONSISTENT_SOURCE


def test_duplicate_names_and_malformed_limits_reject():
    duplicate = f'<robot name="duplicate">{_link("same")}{_link("same")}</robot>'
    with pytest.raises(URDFImportError) as duplicate_error:
        parse_urdf_text(duplicate)
    assert duplicate_error.value.status == AdapterStatus.INCONSISTENT_SOURCE

    malformed_limit = _TWO_LINK.replace('lower="-1"', 'lower="2"')
    with pytest.raises(URDFImportError) as limit_error:
        parse_urdf_text(malformed_limit)
    assert limit_error.value.status == AdapterStatus.MALFORMED_SOURCE

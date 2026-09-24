#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.axes import AxisKey
from phydrax.dynamics import InputLayout, StateLayout
from phydrax.linalg import ArraySpace, DualSpace


_ROLES = ("point", "local", "tangent", "local_cotangent", "cotangent")


def _layout(**overrides) -> StateLayout:
    declaration = {
        "axes": ("site", "field"),
        "component_names": ("a0", "b0", "a1", "b1"),
        "layout_id": "pendulum-pair",
    }
    declaration.update(overrides)
    return StateLayout((2, 2), **declaration)


def test_point_port_carries_layout_storage_axes_and_geometry():
    layout = _layout()
    port = layout.value_port()

    assert port.semantic_id == "pendulum-pair:point"
    assert port.event_shape == (2, 2)
    assert port.component_ids == ("a0", "b0", "a1", "b1")
    assert port.axis_keys == (
        AxisKey("state-layout:pendulum-pair", "site"),
        AxisKey("state-layout:pendulum-pair", "field"),
    )
    assert port.space_id == layout.geometry.geometry_id
    assert port.dimensions is None
    assert port.frame_id is None
    assert port.normalization_id is None
    assert port.variance == "neutral"
    assert port.representation == "state-point"


def test_differential_ports_use_declared_role_spaces_and_variance():
    local_space = ArraySpace((3,), space_id="chart")
    tangent_space = ArraySpace((3,), space_id="velocity")
    layout = StateLayout(
        (4,),
        component_names=("qw", "qx", "qy", "qz"),
        local_space=local_space,
        tangent_space=tangent_space,
        local_component_names=("rx", "ry", "rz"),
        tangent_component_names=("wx", "wy", "wz"),
        layout_id="attitude",
    )
    local = layout.value_port(role="local")
    tangent = layout.value_port(role="tangent")
    local_cotangent = layout.value_port(role="local_cotangent")
    cotangent = layout.value_port(role="cotangent")

    assert (local.semantic_id, local.space_id) == ("attitude:local", "chart")
    assert (tangent.semantic_id, tangent.space_id) == ("attitude:tangent", "velocity")
    assert local.event_shape == tangent.event_shape == (3,)
    assert local.component_ids == local_cotangent.component_ids == ("rx", "ry", "rz")
    assert tangent.component_ids == cotangent.component_ids == ("wx", "wy", "wz")
    assert local.variance == tangent.variance == "contravariant"
    assert local_cotangent.variance == cotangent.variance == "covariant"
    assert local_cotangent.space_id == DualSpace(local_space).space_id
    assert cotangent.space_id == DualSpace(tangent_space).space_id
    assert local.axis_keys is None and local.dimensions is None
    assert local.representation == "space-coordinates"


def test_every_state_role_has_a_distinct_port_even_with_default_spaces():
    layout = StateLayout((3,))
    identifiers = {layout.value_port(role=role).port_id for role in _ROLES}

    assert len(identifiers) == len(_ROLES)
    assert layout.value_port(role="local").port_id != (
        layout.value_port(role="tangent").port_id
    )


def test_state_ports_are_deterministic_and_follow_layout_declarations():
    port = _layout().value_port()

    assert _layout().value_port().port_id == port.port_id
    assert StateLayout((2, 2)).value_port().port_id == (
        StateLayout((2, 2)).value_port().port_id
    )
    assert _layout(layout_id="other").value_port().port_id != port.port_id
    assert _layout(axes=("field", "site")).value_port().port_id != port.port_id
    assert (
        _layout(component_names=("p", "q", "r", "s")).value_port().port_id != port.port_id
    )


def test_scalar_state_point_port_has_one_component_and_no_axes():
    port = StateLayout((), layout_id="energy").value_port()

    assert port.event_shape == ()
    assert port.component_ids == ("x",)
    assert port.axis_keys == ()


def test_unknown_state_role_is_rejected():
    with pytest.raises(ValueError, match="role must be"):
        _layout().value_port(role="velocity")  # type: ignore[arg-type]


def test_input_port_carries_layout_components_and_axes():
    layout = InputLayout(
        (2,),
        axes=("actuator",),
        component_names=("thrust", "torque"),
        roles=("control", "forcing"),
        layout_id="drive",
    )
    port = layout.value_port()

    assert port.semantic_id == "drive"
    assert port.event_shape == (2,)
    assert port.component_ids == ("thrust", "torque")
    assert port.axis_keys == (AxisKey("input-layout:drive", "actuator"),)
    assert port.dimensions is None
    assert (port.space_id, port.frame_id, port.normalization_id) == (None, None, None)
    assert port.variance == "neutral"
    assert port.representation == "input-array"


def test_input_port_identity_follows_declared_roles_and_layout():
    control = InputLayout((2,), roles="control").value_port()

    assert InputLayout((2,), roles="control").value_port().port_id == control.port_id
    assert InputLayout((2,), roles="parameter").value_port().port_id != (control.port_id)
    assert InputLayout((3,)).value_port().port_id != control.port_id

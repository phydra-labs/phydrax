#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.axes import AxisKey
from phydrax.discretization import (
    BlockDofLayout,
    DiscreteFieldSpace,
    EntityDofLayout,
    ModalDofLayout,
    TensorDofLayout,
)
from phydrax.linalg import ArraySpace, DualSpace


def _field(name, layout, *, representation="point_value", dual=False):
    space = ArraySpace((layout.size,))
    return DiscreteFieldSpace(
        name,
        "support",
        layout,
        DualSpace(space) if dual else space,
        representation=representation,
    )


def test_tensor_field_port_uses_space_identity_and_named_layout_axes():
    layout = TensorDofLayout(("i", "j"), (2, 3))
    field = _field("u", layout)
    port = field.value_port()

    assert port.semantic_id == "u"
    assert port.space_id == field.field_space_id
    assert port.event_shape == (2, 3)
    assert port.component_ids == tuple(f"u[{i}]" for i in range(6))
    assert port.axis_keys == (
        AxisKey(f"dof-layout:{layout.layout_id}", "i"),
        AxisKey(f"dof-layout:{layout.layout_id}", "j"),
    )
    assert port.representation == "point_value"
    assert port.variance == "primal"
    assert port.dimensions is None
    assert (port.frame_id, port.normalization_id) == (None, None)


def test_component_axes_leave_axis_keys_undeclared():
    port = _field("v", TensorDofLayout(("i",), (2,), component_shape=(3,))).value_port()

    assert port.event_shape == (2, 3)
    assert len(port.component_ids) == 6
    assert port.axis_keys is None


def test_entity_modal_and_block_layout_event_structure():
    entity = _field(
        "flux",
        EntityDofLayout("faces", 4, 4, component_shape=(2,)),
        representation="flux_moment",
    ).value_port()
    assert entity.event_shape == (4, 2)
    assert entity.component_ids == tuple(f"flux[{i}]" for i in range(8))

    modal = _field(
        "a", ModalDofLayout(("k0", "k1")), representation="modal_coefficient"
    ).value_port()
    assert modal.event_shape == (2,)
    assert modal.component_ids == ("k0", "k1")

    vector_modal = _field(
        "b",
        ModalDofLayout(("k0", "k1"), component_shape=(2,)),
        representation="modal_coefficient",
    ).value_port()
    assert vector_modal.component_ids == ("k0[0]", "k0[1]", "k1[0]", "k1[1]")

    block = _field(
        "w",
        BlockDofLayout(
            ("vertices", "cells"),
            (TensorDofLayout(("n",), (3,)), EntityDofLayout("cells", 2, 2)),
        ),
        representation="functional",
    ).value_port()
    assert block.event_shape == (5,)
    assert block.component_ids == tuple(f"w[{i}]" for i in range(5))
    assert block.axis_keys is None


def test_dual_vector_space_declares_dual_variance():
    layout = TensorDofLayout(("i",), (3,))

    primal = _field("r", layout).value_port()
    dual = _field("r", layout, dual=True).value_port()

    assert dual.variance == "dual"
    assert primal.port_id != dual.port_id


def test_ports_are_deterministic_and_distinguish_declared_identity():
    layout = TensorDofLayout(("i",), (3,))

    port = _field("u", layout).value_port()

    assert port.port_id == _field("u", TensorDofLayout(("i",), (3,))).value_port().port_id
    assert port.port_id != _field("p", layout).value_port().port_id
    assert (
        port.port_id
        != _field("u", layout, representation="cell_average").value_port().port_id
    )
    assert port.port_id != _field("u", TensorDofLayout(("k",), (3,))).value_port().port_id


def test_field_without_coefficients_has_no_port():
    with pytest.raises(ValueError, match="'empty'"):
        _field("empty", EntityDofLayout("cells", 0, 0)).value_port()

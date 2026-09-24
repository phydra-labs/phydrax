#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.axes import AxisKey
from phydrax.domain import DatasetDomain, GeometryDomain, ScalarInterval, TimeInterval


def _square(label: str) -> GeometryDomain:
    return GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile(), label=label
    )


def test_domain_ports_follow_label_order_with_label_scoped_components():
    x_port, t_port = (_square("x") @ TimeInterval(0.0, 1.0)).value_ports()

    assert x_port.semantic_id == "x"
    assert x_port.event_shape == (2,)
    assert x_port.component_ids == ("x[0]", "x[1]")
    assert x_port.axis_keys == (AxisKey("domain-label:x", "0"),)
    assert x_port.representation == "domain-coordinates"
    assert x_port.variance == "neutral"
    assert x_port.dimensions is None
    assert (x_port.space_id, x_port.frame_id, x_port.normalization_id) == (
        None,
        None,
        None,
    )

    assert t_port.semantic_id == "t"
    assert t_port.event_shape == ()
    assert t_port.component_ids == ("t",)
    assert t_port.axis_keys == ()


def test_equal_coordinate_schemas_under_different_labels_have_distinct_ports():
    (x_port,) = _square("x").value_ports()
    (y_port,) = _square("y").value_ports()
    (relabeled,) = _square("x").relabel("y").value_ports()

    assert y_port.component_ids == ("y[0]", "y[1]")
    assert y_port.axis_keys == (AxisKey("domain-label:y", "0"),)
    assert x_port.port_id != y_port.port_id
    assert relabeled.port_id == y_port.port_id


def test_coordinate_ports_depend_on_declaration_not_support_values():
    (first,) = ScalarInterval(0.0, 1.0, label="s").value_ports()
    (second,) = ScalarInterval(-3.0, 5.0, label="s").value_ports()

    assert first.component_ids == ("s",)
    assert first.port_id == second.port_id


def test_structured_coordinates_have_no_dense_port():
    domain = DatasetDomain({"u": jnp.zeros((4, 2))}, label="data")

    with pytest.raises(ValueError, match="'data'"):
        domain.value_ports()

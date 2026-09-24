# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import pytest

from phydrax.applications.geophysics import GeophysicalFieldBinding, GeophysicalQuantity
from phydrax.dynamics import StateLayout
from phydrax.nn.operator import (
    OperatorFieldSpec,
    OperatorProblemSpec,
    OperatorQuerySpec,
    OperatorTask,
)
from phydrax.units import (
    derived_unit,
    KELVIN,
    KILOPASCAL,
    METER,
    PASCAL,
    PRESSURE,
    SECOND,
    TEMPERATURE,
    VELOCITY,
)


_LAYOUT = StateLayout((4,), component_names=("temperature", "pressure", "u", "v"))


def _binding(quantity, components):
    return GeophysicalFieldBinding(quantity, state_layout=_LAYOUT, components=components)


def test_state_binding_port_carries_quantity_identity_and_dimension():
    quantity = GeophysicalQuantity(
        "surface_pressure", "surface_pressure", KILOPASCAL, axes=("time", "cell")
    )
    port = _binding(quantity, ("pressure",)).value_port()
    assert port.semantic_id == f"geophysical-quantity:{quantity.quantity_id}"
    assert port.event_shape == (1,)
    assert port.component_ids == ("pressure",)
    assert port.dimensions == (PRESSURE,)
    assert port.representation == "geophysical-field"
    assert port.variance == "neutral"
    assert port.axis_keys is None


def test_multi_component_binding_keeps_selected_component_order():
    velocity = GeophysicalQuantity(
        "wind", "velocity", derived_unit("m/s", ((METER, 1), (SECOND, -1)))
    )
    port = _binding(velocity, ("v", "u")).value_port()
    assert port.event_shape == (2,)
    assert port.component_ids == ("v", "u")
    assert port.dimensions == (VELOCITY, VELOCITY)


def test_port_identity_follows_quantity_declaration():
    temperature = GeophysicalQuantity("temperature", "temperature", KELVIN)
    port = _binding(temperature, ("temperature",)).value_port()
    assert port.dimensions == (TEMPERATURE,)
    assert _binding(temperature, ("temperature",)).value_port().port_id == port.port_id
    anomaly = GeophysicalQuantity(
        "temperature", "temperature", KELVIN, reference_configuration="anomaly"
    )
    assert _binding(anomaly, ("temperature",)).value_port().port_id != port.port_id
    kilopascal = GeophysicalQuantity("p", "pressure", KILOPASCAL)
    pascal = GeophysicalQuantity("p", "pressure", PASCAL)
    kilopascal_port = _binding(kilopascal, ("pressure",)).value_port()
    pascal_port = _binding(pascal, ("pressure",)).value_port()
    assert kilopascal_port.dimensions == pascal_port.dimensions
    assert kilopascal_port.semantic_id != pascal_port.semantic_id


def test_bindings_without_declared_components_fail_closed():
    task = OperatorTask(
        "temperature-task",
        fields=(
            OperatorFieldSpec("state", role="source", dimension=TEMPERATURE),
            OperatorFieldSpec(
                "temperature", role="target", query_name="column", dimension=TEMPERATURE
            ),
        ),
        dimension_basis=("temperature",),
        queries=(
            OperatorQuerySpec(
                "column", geometry_kind="point_cloud", coordinate_components=("level",)
            ),
        ),
        problem=OperatorProblemSpec(
            source_query_relation="coincident", query_is_fixed=False
        ),
    )
    quantity = GeophysicalQuantity("temperature", "temperature", KELVIN)
    binding = GeophysicalFieldBinding(
        quantity, operator_task=task, field_name="temperature"
    )
    with pytest.raises(ValueError, match="operator-task"):
        binding.value_port()

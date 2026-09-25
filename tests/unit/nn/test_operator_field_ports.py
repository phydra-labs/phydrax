#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

import phydrax as phx
from phydrax.discretization import CochainFieldSpec
from phydrax.nn.operator import (
    OperatorClassificationSpec,
    OperatorFieldSpec,
    OperatorOutputSpec,
    OperatorQuerySpec,
)
from phydrax.nn.operator.representations import (
    CliffordGradeRepresentation,
    TensorFieldBlock,
    TensorFieldLayout,
    TensorType,
)


def _velocity(**overrides):
    declaration = {
        "channels": 3,
        "representation": "vector",
        "component_names": ("u", "v", "w"),
        "dimension": phx.units.VELOCITY,
        "scale": 2.0,
        "offset": 0.0,
    }
    declaration.update(overrides)
    return OperatorFieldSpec("velocity", **declaration)


def test_vector_field_port_carries_declared_semantics():
    port = _velocity().value_port()

    assert port.semantic_id == "velocity"
    assert port.event_shape == (3,)
    assert port.component_ids == ("u", "v", "w")
    assert port.dimensions == (phx.units.VELOCITY,) * 3
    assert port.representation == "vector"
    assert port.variance == "contravariant"
    assert port.space_id is None
    assert port.frame_id is None
    assert port.axis_keys is None
    assert port.normalization_id is not None


def test_field_port_is_deterministic_across_equal_and_serialized_declarations():
    field = _velocity()
    restored = OperatorFieldSpec.from_dict(field.to_dict())

    assert field.value_port().port_id == _velocity().value_port().port_id
    assert restored.value_port().port_id == field.value_port().port_id


def test_unnamed_components_follow_scalar_and_indexed_rules():
    scalar = OperatorFieldSpec("pressure", dimension=phx.units.PRESSURE).value_port()
    named_scalar = OperatorFieldSpec("pressure", component_names=("p",)).value_port()
    channels = OperatorFieldSpec("q", channels=2).value_port()
    single = OperatorFieldSpec("q", channels=1).value_port()

    assert (scalar.event_shape, scalar.component_ids) == ((), ("pressure",))
    assert scalar.dimensions == (phx.units.PRESSURE,)
    assert scalar.variance == "neutral"
    assert named_scalar.component_ids == ("p",)
    assert (channels.event_shape, channels.component_ids) == ((2,), ("q[0]", "q[1]"))
    assert channels.representation == "generic_channels"
    assert (single.event_shape, single.component_ids) == ((1,), ("q[0]",))


def test_covector_field_port_is_covariant():
    port = _velocity(representation="covector").value_port()

    assert port.variance == "covariant"


@pytest.mark.parametrize(
    "overrides",
    [
        {"scale": 3.0},
        {"offset": 1.0},
        {"scale": (2.0, 2.0, 4.0)},
        {"dimension": phx.units.LENGTH},
        {"component_names": ("x", "y", "z")},
        {"representation": "generic_channels"},
        {
            "cochain": CochainFieldSpec(
                1, cell_orientation="signed", sampling="cell_integral"
            )
        },
    ],
)
def test_differing_field_declarations_change_port_id(overrides):
    assert _velocity(**overrides).value_port().port_id != _velocity().value_port().port_id


def test_cochain_declaration_identifies_the_field_space():
    def flux(degree: int, side: str = "primal") -> OperatorFieldSpec:
        return OperatorFieldSpec(
            "flux",
            cochain=CochainFieldSpec(
                degree,
                complex_side=side,
                cell_orientation="signed",
                sampling="cell_integral",
            ),
        )

    edge = flux(1).value_port()

    assert edge.space_id is not None
    assert edge.space_id == flux(1).value_port().space_id
    assert edge.space_id != flux(2).value_port().space_id
    assert edge.space_id != flux(1, "dual").value_port().space_id
    assert edge.variance == "neutral"


def test_tensor_and_clifford_fields_use_the_packed_channel_event():
    vector = TensorType(("contravariant",), dimension=2)
    stress = TensorType(("covariant", "covariant"), dimension=2)
    layout = TensorFieldLayout(
        (TensorFieldBlock("velocity", vector), TensorFieldBlock("stress", stress))
    )
    swapped = TensorFieldLayout(
        (TensorFieldBlock("stress", stress), TensorFieldBlock("velocity", vector))
    )
    tensor = OperatorFieldSpec("state", channels=6, tensor_layout=layout).value_port()
    reordered = OperatorFieldSpec("state", channels=6, tensor_layout=swapped).value_port()
    algebra = phx.metrix.clifford.CliffordAlgebraSpec((1, 1))
    clifford = OperatorFieldSpec(
        "multivector",
        channels=4,
        clifford_layout=CliffordGradeRepresentation(algebra, (1, 1, 1)),
    ).value_port()

    assert tensor.event_shape == (6,)
    assert tensor.component_ids == tuple(f"state[{index}]" for index in range(6))
    assert tensor.representation == "tensor"
    assert tensor.variance == "neutral"
    assert tensor.space_id is not None
    assert tensor.space_id != reordered.space_id
    assert clifford.event_shape == (4,)
    assert clifford.representation == "clifford_multivector"
    assert clifford.space_id is not None


def test_classification_field_port_is_dimensionless_scalar():
    classification = OperatorClassificationSpec("binary", ("off", "on"))
    field = OperatorFieldSpec(
        "phase",
        role="target",
        output_spec=OperatorOutputSpec("scalar", classification=classification),
    )
    port = field.value_port()

    assert port.event_shape == ()
    assert port.component_ids == ("phase",)
    assert port.dimensions == (phx.units.DIMENSIONLESS,)


def _query(**overrides):
    declaration = {
        "geometry_kind": "point_cloud",
        "coordinate_components": ("x", "t"),
        "coordinate_dimensions": (phx.units.LENGTH, phx.units.TIME),
    }
    declaration.update(overrides)
    return OperatorQuerySpec("points", **declaration)


def test_query_port_carries_coordinate_components_and_dimensions():
    port = _query().value_port()
    undeclared = _query(coordinate_dimensions=()).value_port()

    assert port.semantic_id == "points"
    assert port.event_shape == (2,)
    assert port.component_ids == ("x", "t")
    assert port.dimensions == (phx.units.LENGTH, phx.units.TIME)
    assert port.representation == "query-coordinates"
    assert port.variance == "neutral"
    assert (port.space_id, port.frame_id, port.normalization_id) == (None, None, None)
    assert port.axis_keys is None
    assert undeclared.dimensions is None
    assert undeclared.port_id != port.port_id


def test_query_port_identity_follows_coordinate_declarations():
    port = _query().value_port()
    restored = OperatorQuerySpec.from_dict(_query().to_dict()).value_port()

    assert restored.port_id == port.port_id
    assert _query(fixed_geometry=True).value_port().port_id == port.port_id
    assert _query(coordinate_components=("y", "t")).value_port().port_id != port.port_id
    assert (
        _query(coordinate_dimensions=(phx.units.LENGTH, phx.units.LENGTH))
        .value_port()
        .port_id
        != port.port_id
    )
    assert (
        OperatorQuerySpec(
            "grid",
            geometry_kind="point_cloud",
            coordinate_components=("x", "t"),
            coordinate_dimensions=(phx.units.LENGTH, phx.units.TIME),
        )
        .value_port()
        .port_id
        != port.port_id
    )

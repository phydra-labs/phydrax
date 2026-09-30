#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import pytest

from phydrax.axes import AxisKey
from phydrax.discretization import (
    BlockDofLayout,
    DiscreteFieldSpace,
    EntityDofLayout,
    FieldConformity,
    ModalDofLayout,
    TensorDofLayout,
)
from phydrax.exterior._form_type import FormType
from phydrax.linalg import ArraySpace, DualSpace


def _field(
    name: Any, layout: Any, *, representation: Any = "point_value", dual: Any = False
) -> Any:
    space = ArraySpace((layout.size,))
    return DiscreteFieldSpace(
        name,
        "support",
        layout,
        DualSpace(space) if dual else space,
        representation=representation,
    )


def test_discrete_space_ports_scenario_1() -> None:
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
    port = _field("v", TensorDofLayout(("i",), (2,), component_shape=(3,))).value_port()

    assert port.event_shape == (2, 3)
    assert len(port.component_ids) == 6
    assert port.axis_keys is None
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


def test_discrete_space_ports_scenario_2() -> None:
    layout = TensorDofLayout(("i",), (3,))

    primal = _field("r", layout).value_port()
    dual = _field("r", layout, dual=True).value_port()

    assert dual.variance == "dual"
    assert primal.port_id != dual.port_id
    layout = TensorDofLayout(("i",), (3,))

    port = _field("u", layout).value_port()

    assert port.port_id == _field("u", TensorDofLayout(("i",), (3,))).value_port().port_id
    assert port.port_id != _field("p", layout).value_port().port_id
    assert (
        port.port_id
        != _field("u", layout, representation="cell_average").value_port().port_id
    )
    assert port.port_id != _field("u", TensorDofLayout(("k",), (3,))).value_port().port_id
    with pytest.raises(ValueError, match="'empty'"):
        _field("empty", EntityDofLayout("cells", 0, 0)).value_port()


def test_undeclared_form_preserves_historical_field_space_identity() -> None:
    field = DiscreteFieldSpace(
        "u",
        "mesh",
        TensorDofLayout(("node",), (3,), layout_id="nodes"),
        ArraySpace((3,), space_id="nodal"),
        representation="point_value",
    )
    assert field.form_type is None
    assert field.field_space_id == (
        "29804d67a326eff8059256b23334061e025ed27c614a0fff9bf2a35d77d41b95"
    )


def test_field_space_identity_distinguishes_declared_form_types() -> None:
    layout = TensorDofLayout(("node",), (3,), layout_id="nodes")
    space = ArraySpace((3,), space_id="nodal")
    plain = DiscreteFieldSpace("u", "mesh", layout, space, representation="cochain")
    scalar = DiscreteFieldSpace(
        "u", "mesh", layout, space, representation="cochain", form_type=FormType(3, 0)
    )
    twisted = DiscreteFieldSpace(
        "u",
        "mesh",
        layout,
        space,
        representation="cochain",
        form_type=FormType(3, 0, twist="twisted"),
    )
    assert scalar.form_type is not None
    assert scalar.form_type.form_type_id == FormType(3, 0).form_type_id
    assert len({plain.field_space_id, scalar.field_space_id, twisted.field_space_id}) == 3


def test_field_space_form_persistence_preserves_binding_identity() -> None:
    form = FormType(2, 1, twist="twisted", fiber_shape=(2,), ambient_dimension=3)
    payload = form.to_dict()
    assert payload == {
        "dimension": 2,
        "degree": 1,
        "twist": "twisted",
        "fiber_shape": [2],
        "ambient_dimension": 3,
    }
    layout = TensorDofLayout(("node",), (3,), layout_id="nodes")
    space = ArraySpace((3,), space_id="nodal")
    original = DiscreteFieldSpace(
        "u", "mesh", layout, space, representation="cochain", form_type=form
    )
    restored = DiscreteFieldSpace(
        "u",
        "mesh",
        layout,
        space,
        representation="cochain",
        form_type=FormType.from_dict(payload),
    )
    assert restored.field_space_id == original.field_space_id


def test_field_space_refuses_obsolete_cochain_conformity() -> None:
    with pytest.raises(ValueError, match="conformity"):
        DiscreteFieldSpace(
            "u",
            "mesh",
            TensorDofLayout(("node",), (3,)),
            ArraySpace((3,)),
            representation="cochain",
            conformity="cochain",  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("conformity", ("H1", "Hcurl", "Hdiv"))
def test_declared_form_degree_refuses_incompatible_vector_proxy_conformity(
    conformity: FieldConformity,
) -> None:
    form_type = FormType(4, 2, twist="untwisted")
    with pytest.raises(ValueError, match="form degree"):
        DiscreteFieldSpace(
            "two-form",
            "four-dimensional-support",
            EntityDofLayout("two-cells", 2, 2),
            ArraySpace((2,), space_id="two-form-coefficients"),
            representation="basis_coefficient",
            conformity=conformity,
            form_type=form_type,
        )


def test_generic_form_conformity_requires_declared_form_type() -> None:
    with pytest.raises(ValueError, match="HLambda.*form_type"):
        DiscreteFieldSpace(
            "two-form",
            "mesh",
            TensorDofLayout(("coefficient",), (6,)),
            ArraySpace((6,)),
            representation="basis_coefficient",
            conformity="HLambda",
        )


@pytest.mark.parametrize("degree", (0, 2, 4))
def test_generic_form_conformity_retains_declared_degree_identity(degree: int) -> None:
    form = FormType(4, degree)
    field = DiscreteFieldSpace(
        "form",
        "mesh",
        TensorDofLayout(("coefficient",), (6,)),
        ArraySpace((6,)),
        representation="basis_coefficient",
        conformity="HLambda",
        form_type=form,
    )
    assert field.conformity == "HLambda"
    assert field.form_type is not None
    assert field.form_type.form_type_id == form.form_type_id

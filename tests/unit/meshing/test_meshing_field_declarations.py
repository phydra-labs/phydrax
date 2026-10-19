#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from copy import copy
from dataclasses import fields, replace
from typing import Any

import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.discretization._spaces import GlobalCoefficientId


@pytest.fixture
def common_field_inputs() -> dict[str, Any]:
    """Declare a scalar coefficient bank with explicit scientific row identities."""
    space_id = "declared-scalar-field"
    coefficient = GlobalCoefficientId(space_id, 0, ())
    bank = phx.lifecycle.MeshingCoefficientIdentityBank(
        space_id, ((0, (), coefficient.coefficient_id),)
    )
    binding = phx.lifecycle.MeshingFieldIndexBinding(
        bank_name="field/global_coefficient_ids",
        source_result_id="accepted-carrier",
        source_topology_id="accepted-topology",
        source_geometry_id="accepted-geometry",
        source_evidence_id=None,
        source_epoch_id=None,
        index_values=np.asarray([0], dtype=np.int64),
        index_axes=(0,),
        field_space_id=space_id,
        coefficient_bank=bank,
    )
    role = phx.lifecycle.MeshingFieldStateRole(
        "state",
        "scalar",
        "coefficients",
        (1,),
        np.dtype(np.float64),
        0,
        index_binding=binding,
    )
    return {
        "part_name": "domain",
        "topology_id": "accepted-topology",
        "geometry_layout_id": "accepted-geometry-layout",
        "field_space_ids": frozendict({"scalar": space_id}),
        "value_units": frozendict({"scalar": (phx.units.ONE,)}),
        "maximum_derivative_orders": frozendict({"scalar": 0}),
        "branch_policy": phx.BranchDifferentiationPolicy.SMOOTH,
        "trace_policy": phx.discretization.FieldTracePolicy("cell-sided"),
        "state_roles": (role,),
        "history_policy": "none",
        "numeric_version": "accepted",
    }


@pytest.fixture
def finite_element_declaration(
    common_field_inputs: dict[str, Any],
) -> phx.lifecycle.MeshingFiniteElementFieldDeclaration:
    element = phx.discretization.lagrange_element("triangle", 1)
    return phx.lifecycle.MeshingFiniteElementFieldDeclaration(
        **common_field_inputs,
        finite_element_fields=(
            phx.discretization.FiniteElementFieldSpec("scalar", element),
        ),
        precision_policy=phx.discretization.FiniteElementPrecisionPolicy(),
    )


@pytest.fixture
def finite_volume_declaration(
    common_field_inputs: dict[str, Any],
) -> phx.lifecycle.MeshingFiniteVolumeFieldDeclaration:
    reconstruction = phx.discretization.PiecewiseConstantReconstruction()
    return phx.lifecycle.MeshingFiniteVolumeFieldDeclaration(
        **common_field_inputs,
        finite_volume_field_name="scalar",
        finite_volume_component_names=("scalar",),
        finite_volume_coordinate_policy="accepted-map",
        reconstruction_policy=reconstruction,
        reconstruction_policy_id=reconstruction.plan_id,
    )


def test_common_field_contract_is_abstract(common_field_inputs: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="abstract"):
        phx.lifecycle.MeshingFieldDeclaration(**common_field_inputs)  # ty: ignore[call-non-callable]


def test_finite_element_declaration_has_no_finite_volume_fields(
    finite_element_declaration: phx.lifecycle.MeshingFiniteElementFieldDeclaration,
) -> None:
    names = {field.name for field in fields(finite_element_declaration)}
    assert finite_element_declaration.owner == "finite_element"
    assert "owner" not in names
    assert "finite_volume_system" not in names
    assert "reconstruction_policy" not in names
    with pytest.raises(TypeError, match="finite_volume_system"):
        replace(
            finite_element_declaration, finite_volume_system=phx.equations.EulerSystem(2)
        )


def test_finite_volume_declaration_has_no_finite_element_fields(
    finite_volume_declaration: phx.lifecycle.MeshingFiniteVolumeFieldDeclaration,
) -> None:
    names = {field.name for field in fields(finite_volume_declaration)}
    assert finite_volume_declaration.owner == "finite_volume"
    assert "owner" not in names
    assert "finite_element_fields" not in names
    assert "precision_policy" not in names
    with pytest.raises(TypeError, match="finite_element_fields"):
        replace(finite_volume_declaration, finite_element_fields=())


@pytest.mark.parametrize(
    "fixture_name",
    ["finite_element_declaration", "finite_volume_declaration"],
    ids=["finite-element", "finite-volume"],
)
def test_stored_declaration_reconstruction_preserves_state_identity(
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    declaration = request.getfixturevalue(fixture_name)
    restored = type(declaration).from_stored_fields(declaration)
    assert type(restored) is type(declaration)
    assert restored.part_name == declaration.part_name
    assert restored.field_space_ids == declaration.field_space_ids
    assert restored.state_roles == declaration.state_roles
    assert restored.source_wall_policy == declaration.source_wall_policy
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(
        declaration
    )


def test_finite_volume_policy_identity_is_revalidated(
    finite_volume_declaration: phx.lifecycle.MeshingFiniteVolumeFieldDeclaration,
) -> None:
    with pytest.raises(ValueError, match="reconstruction_policy_id"):
        replace(finite_volume_declaration, reconstruction_policy_id="foreign-policy")


@pytest.mark.parametrize(
    "fixture_name",
    ["finite_element_declaration", "finite_volume_declaration"],
    ids=["finite-element", "finite-volume"],
)
def test_field_derivative_ceiling_is_revalidated(
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    declaration = request.getfixturevalue(fixture_name)
    with pytest.raises(ValueError, match="maximum_derivative_orders"):
        replace(declaration, maximum_derivative_orders=frozendict({"scalar": 2}))


def test_material_history_requires_a_material_history_role(
    finite_element_declaration: phx.lifecycle.MeshingFiniteElementFieldDeclaration,
) -> None:
    with pytest.raises(ValueError, match="state_roles"):
        replace(finite_element_declaration, history_policy="material-history")


def test_field_units_cover_every_declared_component(
    finite_volume_declaration: phx.lifecycle.MeshingFiniteVolumeFieldDeclaration,
) -> None:
    with pytest.raises(ValueError, match="value_units"):
        replace(finite_volume_declaration, value_units=frozendict())


@pytest.mark.parametrize(
    "fixture_name",
    ["finite_element_declaration", "finite_volume_declaration"],
    ids=["finite-element", "finite-volume"],
)
def test_stored_declaration_rejects_corrupted_field_semantics(
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    declaration = copy(request.getfixturevalue(fixture_name))
    object.__setattr__(
        declaration, "maximum_derivative_orders", frozendict({"scalar": 2})
    )
    with pytest.raises(ValueError, match="maximum_derivative_orders"):
        type(declaration).from_stored_fields(declaration)


def test_stored_declaration_rejects_extra_persisted_fields(
    finite_element_declaration: phx.lifecycle.MeshingFiniteElementFieldDeclaration,
) -> None:
    declaration = copy(finite_element_declaration)
    object.__setattr__(declaration, "undeclared_field", 1)
    with pytest.raises(ValueError, match="field set"):
        type(declaration).from_stored_fields(declaration)


def test_stored_declaration_cannot_change_physical_owner(
    finite_volume_declaration: phx.lifecycle.MeshingFiniteVolumeFieldDeclaration,
) -> None:
    with pytest.raises(TypeError, match="registered declaration type"):
        phx.lifecycle.MeshingFiniteElementFieldDeclaration.from_stored_fields(
            finite_volume_declaration
        )


@pytest.mark.parametrize("record_kind", ["role", "index", "coefficient-bank"])
def test_stored_field_metadata_round_trip(
    common_field_inputs: dict[str, Any],
    record_kind: str,
) -> None:
    role = common_field_inputs["state_roles"][0]
    records = {
        "role": role,
        "index": role.index_binding,
        "coefficient-bank": role.index_binding.coefficient_bank,
    }
    record = records[record_kind]
    restored = type(record).from_stored_fields(record)
    assert type(restored) is type(record)
    assert restored == record


@pytest.mark.parametrize(
    ("record_kind", "field", "value", "error"),
    [
        pytest.param("role", "state_epoch", -1, ValueError, id="state-epoch"),
        pytest.param(
            "index",
            "index_values",
            np.asarray([0], dtype=np.int32),
            TypeError,
            id="index-dtype",
        ),
        pytest.param(
            "coefficient-bank",
            "field_space_id",
            "",
            ValueError,
            id="coefficient-space",
        ),
    ],
)
def test_stored_field_metadata_rejects_corruption(
    common_field_inputs: dict[str, Any],
    record_kind: str,
    field: str,
    value: Any,
    error: type[Exception],
) -> None:
    role = common_field_inputs["state_roles"][0]
    records = {
        "role": role,
        "index": role.index_binding,
        "coefficient-bank": role.index_binding.coefficient_bank,
    }
    record = copy(records[record_kind])
    object.__setattr__(record, field, value)
    with pytest.raises(error, match=field):
        type(record).from_stored_fields(record)

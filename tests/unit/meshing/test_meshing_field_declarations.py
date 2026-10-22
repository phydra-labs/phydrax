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
from phydrax._sampling import derive_key
from phydrax.lifecycle._meshing_field_records import (
    MeshingFieldIndexBinding,
    MeshingFieldStateRole,
    MeshingRandomStream,
    rebuild_meshing_field_record,
)


@pytest.fixture
def common_field_inputs() -> dict[str, Any]:
    """Declare a scalar coefficient bank with its derived scientific row identity."""
    space_id = "declared-scalar-field"
    bank = phx.lifecycle.MeshingCoefficientIdentityBank.from_rows(space_id, ((0, ()),))
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


@pytest.fixture
def random_stream() -> MeshingRandomStream:
    return MeshingRandomStream(
        "root-key",
        "threefry2x32",
        "meshing-test",
        "cell-probe",
        ("scalar",),
        "rademacher",
    )


@pytest.fixture
def random_counter_role(random_stream: MeshingRandomStream) -> MeshingFieldStateRole:
    binding = MeshingFieldIndexBinding(
        bank_name="epoch/cell_ids",
        source_result_id="accepted-carrier",
        source_topology_id="accepted-topology",
        source_geometry_id="accepted-geometry",
        source_evidence_id="accepted-evidence",
        source_epoch_id="accepted-epoch",
        index_values=np.asarray([[1, 2**32 + 1, -1]], dtype=np.int64),
        index_axes=(0, 1),
    )
    return MeshingFieldStateRole(
        "raw-counter",
        "scalar",
        "rng-counter",
        binding.index_values.shape,
        np.dtype(np.uint64),
        0,
        index_binding=binding,
        random_stream=random_stream,
    )


def test_entity_random_keys_fold_both_id_and_counter_words(
    random_stream: MeshingRandomStream,
) -> None:
    root = jax.random.key(37, impl=random_stream.implementation)
    data = jax.random.key_data(root)
    ids = np.asarray([1, 2**32 + 1, 2**63 - 1, 1], dtype=np.int64)
    counters = np.asarray([7, 7, 2**64 - 1, 2**32 + 7], dtype=np.uint64)
    actual = random_stream.entity_keys(data, ids, counters)
    expected = [
        derive_key(
            root,
            random_stream.address,
            int(identifier) >> 32,
            int(identifier) & 0xFFFFFFFF,
            int(counter) >> 32,
            int(counter) & 0xFFFFFFFF,
        )
        for identifier, counter in zip(ids, counters, strict=True)
    ]
    np.testing.assert_array_equal(
        jax.random.key_data(actual),
        np.stack([jax.random.key_data(key) for key in expected]),
    )
    assert np.unique(np.asarray(jax.random.key_data(actual)), axis=0).shape[0] == 4
    # A changed owner order and batch extent must not alter an entity's next draw.
    order = np.asarray([3, 1, 0, 2], dtype=np.int64)
    reordered = random_stream.entity_keys(data, ids[order], counters[order])
    draws = jax.vmap(lambda key: jax.random.bits(key, (8,)))(actual)
    np.testing.assert_array_equal(
        jax.vmap(lambda key: jax.random.bits(key, (8,)))(reordered), draws[order]
    )
    for row in range(ids.size):
        singleton = random_stream.entity_keys(
            data, ids[row : row + 1], counters[row : row + 1]
        )
        np.testing.assert_array_equal(jax.random.bits(singleton[0], (8,)), draws[row])


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_random_counter_role_round_trip_preserves_unsigned_schema(
    random_counter_role: MeshingFieldStateRole,
    dtype: type[np.uint32] | type[np.uint64],
) -> None:
    role = replace(random_counter_role, dtype=np.dtype(dtype))
    restored = rebuild_meshing_field_record(role)
    assert restored == role
    assert rebuild_meshing_field_record(role.random_stream) == role.random_stream


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_, np.uint16])
def test_random_counter_role_refuses_non_counter_dtypes(
    random_counter_role: MeshingFieldStateRole,
    dtype: type[np.generic],
) -> None:
    with pytest.raises(ValueError, match="uint32 or uint64"):
        replace(random_counter_role, dtype=np.dtype(dtype))


def test_random_roles_require_unitless_addressed_entity_state(
    random_counter_role: MeshingFieldStateRole,
    common_field_inputs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="unitless"):
        replace(random_counter_role, value_units=(phx.units.ONE,))
    with pytest.raises(TypeError, match="random_stream"):
        replace(random_counter_role, random_stream=None)
    with pytest.raises(ValueError, match="stable entity ID"):
        replace(
            random_counter_role,
            shape=(1,),
            index_binding=common_field_inputs["state_roles"][0].index_binding,
        )
    with pytest.raises(ValueError, match="component axes"):
        replace(random_counter_role, shape=(*random_counter_role.shape, 1))
    with pytest.raises(ValueError, match="absent for physical state"):
        replace(random_counter_role, role="stage-history")


def test_random_counter_declaration_requires_its_persisted_root_key(
    random_counter_role: MeshingFieldStateRole,
    random_stream: MeshingRandomStream,
    finite_element_declaration: phx.lifecycle.MeshingFiniteElementFieldDeclaration,
) -> None:
    coefficient = finite_element_declaration.state_roles[0]
    key = MeshingFieldStateRole(
        random_stream.key_entry_name,
        "scalar",
        "rng-key",
        random_stream.key_shape,
        np.dtype(np.uint32),
        0,
        random_stream=random_stream,
    )
    with pytest.raises(ValueError, match="missing or foreign root key"):
        replace(
            finite_element_declaration,
            state_roles=(coefficient, random_counter_role),
            history_policy="accepted-state",
        )
    declaration = replace(
        finite_element_declaration,
        state_roles=(coefficient, random_counter_role, key),
        history_policy="accepted-state",
    )
    restored = type(declaration).from_stored_fields(declaration)
    assert restored.state_roles == declaration.state_roles
    with pytest.raises(ValueError, match="missing or foreign root key"):
        replace(
            declaration,
            state_roles=(
                coefficient,
                random_counter_role,
                replace(key, random_stream=replace(random_stream, purpose="foreign")),
            ),
        )
    with pytest.raises(ValueError, match="rng-key schema"):
        replace(key, dtype=np.dtype(np.int32))


def test_random_key_address_rejects_truncated_or_signed_inputs(
    random_stream: MeshingRandomStream,
) -> None:
    data = jax.random.key_data(jax.random.key(1))
    with pytest.raises(ValueError, match="incompatible rows"):
        random_stream.entity_keys(
            data, np.asarray([1], dtype=np.int32), np.asarray([1], dtype=np.uint32)
        )
    with pytest.raises(ValueError, match="incompatible rows"):
        random_stream.entity_keys(
            data, np.asarray([1], dtype=np.int64), np.asarray([1], dtype=np.int64)
        )
    with pytest.raises(ValueError, match="negative identity"):
        random_stream.entity_keys(
            data, np.asarray([-1], dtype=np.int64), np.asarray([1], dtype=np.uint32)
        )


def test_euler_history_preserves_four_component_axes_and_physical_units(
    random_counter_role: MeshingFieldStateRole,
    finite_element_declaration: phx.lifecycle.MeshingFiniteElementFieldDeclaration,
) -> None:
    from tools.meshing_qualification.scenarios._distributed_euler import state_units

    units = state_units()
    role = replace(
        random_counter_role,
        entry_name="allocated-euler-conserved",
        role="stage-history",
        shape=(*random_counter_role.shape, 4),
        dtype=np.dtype(np.float64),
        value_units=units,
        random_stream=None,
    )
    declaration = replace(
        finite_element_declaration,
        state_roles=(*finite_element_declaration.state_roles, role),
        history_policy="accepted-state",
    )
    restored = type(declaration).from_stored_fields(declaration)
    assert restored.state_roles[-1] == role
    assert role.shape[-1] == len(units) == 4
    assert units[0] != units[1] == units[2] != units[3]

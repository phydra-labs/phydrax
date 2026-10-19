"""Association label banks preserve ordered source identities and restore bindings."""

from dataclasses import fields, replace
from pathlib import Path

import jax
import numpy as np
import pytest

from phydrax._model._structure import (
    model_from_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.lifecycle._meshing_sources import (
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
)
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationSourceBank,
    GeometrySourceEntityRole,
)
from phydrax.meshing._host_banks import entity_key
from tests._support.recipes import recipe_field_position


def _association() -> GeometryAssociation:
    return GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        "label-bank-source",
        "authored-r1",
        "target-entity-set",
        np.arange(4, dtype=np.int64),
        tuple(entity_key("authored-r1", "region", row) for row in (0, 1, 0, 1)),
        np.zeros(4, dtype=np.float64),
        resolved=np.ones(4, dtype=np.bool_),
        ambiguous=np.zeros(4, dtype=np.bool_),
        source_dimensions=np.full(4, 3, dtype=np.int8),
        source_indices=np.asarray((0, 1, 0, 1), dtype=np.int64),
        source_entity_roles=(GeometrySourceEntityRole.REGION,) * 4,
    )


def test_source_bank_retains_all_rows_and_only_one_copy_of_each_identity() -> None:
    association = _association()
    bank = association.source_entities
    assert bank.row_indices == (0, 1, 0, 1)
    assert len(bank.entries) == 2
    assert bank.entity_ids() == tuple(
        entity_key("authored-r1", "region", row) for row in (0, 1, 0, 1)
    )
    assert bank.roles() == (GeometrySourceEntityRole.REGION,) * 4
    assert bank.occurrence_paths() == ((),) * 4
    assert bank.entity_id(np.int64(1)) == entity_key("authored-r1", "region", 1)
    assert bank.role(1) is GeometrySourceEntityRole.REGION
    assert bank.occurrence_path(1) == ()
    leaves = jax.tree_util.tree_leaves(association)
    assert len(leaves) == 10
    assert all(isinstance(value, jax.Array) for value in leaves)
    assert "source_entities" in tuple(field.name for field in fields(association))
    assert not {"source_entity_ids", "source_entity_roles", "source_occurrence_paths"} & {
        field.name for field in fields(association)
    }


def test_source_bank_keeps_occurrences_and_explicit_roles_distinct() -> None:
    bank = GeometryAssociationSourceBank.from_rows(
        ("entity", "entity", "entity", "entity"),
        (
            GeometrySourceEntityRole.FACET,
            GeometrySourceEntityRole.FACET,
            GeometrySourceEntityRole.REGION,
            GeometrySourceEntityRole.FACET,
        ),
        (("left",), ("right",), ("left",), ("left",)),
    )
    assert len(bank.entries) == 3
    assert bank.row_indices[0] == bank.row_indices[3]
    assert len(set(bank.row_indices[:3])) == 3
    assert bank.occurrence_paths() == (("left",), ("right",), ("left",), ("left",))
    assert bank.roles() == (
        GeometrySourceEntityRole.FACET,
        GeometrySourceEntityRole.FACET,
        GeometrySourceEntityRole.REGION,
        GeometrySourceEntityRole.FACET,
    )


def test_source_bank_preserves_absent_versus_declared_unresolved_roles() -> None:
    absent = GeometryAssociationSourceBank.from_rows(("unclassified",), None, ((),))
    declared = GeometryAssociationSourceBank.from_rows(("unclassified",), (None,), ((),))
    assert absent.entries == declared.entries
    assert absent.roles() is None
    assert declared.roles() == (None,)
    assert absent != declared


@pytest.mark.parametrize(
    "malformation",
    (
        "negative",
        "foreign",
        "boolean",
        "unused",
        "duplicate",
        "unordered",
        "mutable-path",
        "untrimmed-name",
        "role-type",
        "undeclared-role",
    ),
)
def test_source_bank_refuses_noncanonical_typed_metadata(malformation: str) -> None:
    bank = GeometryAssociationSourceBank.from_rows(
        ("a", "b"),
        (GeometrySourceEntityRole.VERTEX, GeometrySourceEntityRole.EDGE),
        ((), ()),
    )
    with pytest.raises((TypeError, ValueError), match="GeometryAssociationSourceBank"):
        match malformation:
            case "negative":
                replace(bank, row_indices=(-1, 1))
            case "foreign":
                replace(bank, row_indices=(0, 2))
            case "boolean":
                replace(bank, row_indices=(False, 1))
            case "unused":
                replace(bank, row_indices=(0, 0))
            case "duplicate":
                replace(bank, entries=(bank.entries[0], bank.entries[0]))
            case "unordered":
                replace(bank, entries=bank.entries[::-1])
            case "mutable-path":
                replace(
                    bank,
                    entries=(("a", GeometrySourceEntityRole.VERTEX, []), bank.entries[1]),
                )
            case "untrimmed-name":
                replace(
                    bank,
                    entries=(
                        (" a", GeometrySourceEntityRole.VERTEX, ()),
                        bank.entries[1],
                    ),
                )
            case "role-type":
                replace(bank, entries=(("a", "vertex", ()), bank.entries[1]))
            case "undeclared-role":
                replace(bank, indexed_roles=False)
            case _:
                raise ValueError(
                    f"source_bank.malformation must name a fixture; got {malformation!r}"
                )


@pytest.mark.parametrize("tamper", ("route", "source"))
def test_restored_association_authenticates_source_bank_rows(tamper: str) -> None:
    association = _association()
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(association)
    arrays = model_recipe_array_values(association, recipe, prefix="bank-tamper")
    bank_recipe = recipe["items"][recipe_field_position(recipe, "source_entities")]
    column = "row_indices" if tamper == "route" else "entity_ids"
    value = next(
        value for key, value in bank_recipe["payload"]["items"] if key["value"] == column
    )
    value["values"] = (
        [1, 0, 1, 0]
        if tamper == "route"
        else ["a-foreign-region", association.source_entities.entries[1][0]]
    )
    restored = model_from_array_recipe(recipe, arrays, prefix="bank-tamper")
    with pytest.raises(
        ValueError, match="source_entity_ids|source_entities|association_id"
    ):
        validate_meshing_source_closure(restored)


def test_source_bank_archive_roundtrip_preserves_semantic_identity(
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    association = _association()
    receipt = write_meshing_source_closure(tmp_path / "source-bank", association)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert restored.association_id == association.association_id
    assert restored.source_entities == association.source_entities
    for before, after in zip(
        jax.tree_util.tree_leaves(association),
        jax.tree_util.tree_leaves(restored),
        strict=True,
    ):
        assert np.asarray(before).tobytes() == np.asarray(after).tobytes()


def test_source_bank_archive_roundtrip_preserves_every_role_member(
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    roles = (*GeometrySourceEntityRole, None, GeometrySourceEntityRole.VERTEX)
    identifiers = tuple(
        "unresolved" if role is None else f"source:{role.value}" for role in roles
    )
    paths = (("left",),) * (len(roles) - 1) + (("right",),)
    bank = GeometryAssociationSourceBank.from_rows(identifiers, roles, paths)
    receipt = write_meshing_source_closure(tmp_path / "every-source-role", bank)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert restored == bank
    assert restored.indexed_roles is True
    assert restored.row_indices == bank.row_indices
    for row, (identifier, role, path) in enumerate(
        zip(identifiers, roles, paths, strict=True)
    ):
        assert restored.entity_id(row) == identifier
        assert restored.role(row) is role
        assert restored.occurrence_path(row) == path


@pytest.mark.parametrize("indexed_roles", (False, True))
def test_source_bank_column_record_preserves_optional_roles_and_occurrences(
    indexed_roles: bool,
) -> None:
    roles = (None, GeometrySourceEntityRole.FACET, None) if indexed_roles else None
    bank = GeometryAssociationSourceBank.from_rows(
        ("left-face", "right-face", "left-face"),
        roles,
        (("assembly", "left"), ("assembly", "right"), ("assembly", "left")),
    )
    record = bank.to_record()
    assert record["entity_ids"] == ("left-face", "right-face")
    assert record["entity_roles"] == (None, "facet" if indexed_roles else None)
    assert record["occurrence_paths"] == (
        ("assembly", "left"),
        ("assembly", "right"),
    )
    assert record["row_indices"] == (0, 1, 0)
    assert record["indexed_roles"] is indexed_roles
    assert GeometryAssociationSourceBank.from_record(record) == bank
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(bank)
    restored = model_from_array_recipe(recipe, {}, prefix="bank-columns")
    assert restored == bank
    assert restored.entity_ids() == bank.entity_ids()
    assert restored.roles() == bank.roles()
    assert restored.occurrence_paths() == bank.occurrence_paths()


@pytest.mark.parametrize(
    "tamper", ("missing", "extra", "mutable", "extent", "role", "unused")
)
def test_source_bank_column_record_refuses_malformed_wire_metadata(tamper: str) -> None:
    record = dict(_association().source_entities.to_record())
    match tamper:
        case "missing":
            del record["entity_roles"]
        case "extra":
            record["unknown"] = ()
        case "mutable":
            record["row_indices"] = [0, 1, 0, 1]
        case "extent":
            record["entity_roles"] = ()
        case "role":
            record["entity_roles"] = ("foreign-role", "region")
        case "unused":
            record["row_indices"] = (0, 0, 0, 0)
        case _:
            raise ValueError(
                f"source_bank.tamper must name a wire mutation; got {tamper!r}"
            )
    with pytest.raises((TypeError, ValueError), match="GeometryAssociationSourceBank"):
        GeometryAssociationSourceBank.from_record(record)


@pytest.mark.parametrize("dimension", (0, 1, 2, 3))
def test_volume_consensus_does_not_classify_exterior_boundary(
    dimension: int,
) -> None:
    from phydrax.discretization import CellMesh
    from phydrax.meshing._association import _volume_stratum_consensus
    from phydrax.meshing._association_routes import boundary_mask

    # Four tetrahedra partition the standard tetrahedron at its interior center.
    mesh = CellMesh.from_tetrahedra(
        np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.25, 0.25, 0.25)),
            dtype=np.float64,
        ),
        np.asarray(
            ((0, 1, 2, 4), (0, 3, 1, 4), (0, 2, 3, 4), (1, 3, 2, 4)),
            dtype=np.int32,
        ),
    )
    regions = _volume_stratum_consensus(mesh, dimension, np.zeros((4,), dtype=np.int64))
    if dimension < 3:
        boundary = boundary_mask(mesh, dimension)
        assert np.any(boundary)
        assert np.all(regions[boundary] == -1)
        assert np.all(regions[~boundary] == 0)
    else:
        assert np.array_equal(regions, np.zeros((4,), dtype=np.int64))
    if dimension == 0:
        assert regions[0] == -1
        assert regions[4] == 0


def test_reference_chart_reuse_binds_current_arrays_and_request_scope() -> None:
    import equinox as eqx

    from phydrax.discretization._cell_geometry import (
        coordinate_lagrange_element,
        RestrictedCellGeometryElement,
    )
    from phydrax.discretization._coordinate_enclosure import coordinate_map_images
    from phydrax.geometry._mapped_reference_coverage import (
        PreparedPolynomialReferenceCharts,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    root = coordinate_lagrange_element("triangle", 2)
    element = RestrictedCellGeometryElement(
        root,
        "triangle",
        np.eye(2, dtype=np.float64) * 0.5,
        np.zeros((2,), dtype=np.float64),
    )
    # Open array leaves can change without changing a declared element ID.
    matrix = np.eye(2, dtype=np.float64) * 0.5
    element = eqx.tree_at(lambda item: item.matrix, element, matrix)
    with coordinate_reference_budget() as budget:
        prepared = PreparedPolynomialReferenceCharts()
        first = prepared.prepare(element, root, budget=budget)
        assert first is not None
        assert prepared.prepare(element, root, budget=budget) is first
        before = coordinate_map_images(first[0], first[1], budget=budget)
        matrix[0, 0] = 0.25
        second = prepared.prepare(element, root, budget=budget)
        assert second is not None and second is not first
        after = coordinate_map_images(second[0], second[1], budget=budget)
        assert after != before
        independent = PreparedPolynomialReferenceCharts().prepare(
            element, root, budget=budget
        )
        assert independent is not None and independent[0] is not second[0]
        assert (
            coordinate_map_images(independent[0], independent[1], budget=budget) == after
        )

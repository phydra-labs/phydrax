"""Association label banks preserve ordered source identities and restore bindings."""

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest

from phydrax._model._structure import (
    model_from_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.lifecycle._meshing_sources import (
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
)
from phydrax.meshing._association import support_budget
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationSourceBank,
    GeometryAssociationSourceIdentities,
    GeometrySourceEntityRole,
)
from phydrax.meshing._host_banks import entity_key
from tests._support.mesh_certificate_fixtures import (
    coordinate_budget as coordinate_budget,
    reference_budget as reference_budget,
)
from tests._support.recipes import recipe_field_position


@pytest.mark.parametrize(
    "maximum_memory_bytes",
    (MeshCertificateLimits().maximum_scratch_bytes, 1_536_000_000),
)
def test_support_budget_preserves_explicit_cold_and_nested_cap(
    maximum_memory_bytes: int,
) -> None:
    with support_budget(
        100, 100, None, maximum_memory_bytes=maximum_memory_bytes
    ) as budget:
        assert budget.maximum_memory_bytes == maximum_memory_bytes
        with support_budget(
            100, 100, budget, maximum_memory_bytes=maximum_memory_bytes
        ) as nested:
            assert nested is budget
            assert nested.ticket is budget.ticket
            nested.reserve(1, maximum_memory_bytes)
        assert budget.work_units == 1
        with pytest.raises(CoordinateEnclosureResourceError) as refusal:
            budget.reserve(1, 1)
        assert refusal.value.limit == maximum_memory_bytes
        assert refusal.value.requested == maximum_memory_bytes + 1
        assert budget.work_units == 1


@pytest.mark.parametrize("maximum_memory_bytes", (0, -1))
def test_support_budget_rejects_nonpositive_cap(maximum_memory_bytes: int) -> None:
    with pytest.raises(ValueError, match="maximum_memory_bytes"):
        with support_budget(
            100, 100, None, maximum_memory_bytes=maximum_memory_bytes
        ):
            pytest.fail("Invalid cap admitted")


def test_support_budget_rejects_mismatched_nested_cap() -> None:
    with support_budget(
        100, 100, None, maximum_memory_bytes=1_536_000_000
    ) as budget:
        with pytest.raises(ValueError, match="match the caller memory limit"):
            with support_budget(
                100,
                100,
                budget,
                maximum_memory_bytes=MeshCertificateLimits().maximum_scratch_bytes,
            ):
                pytest.fail("Replacement cap admitted")
        assert budget.work_units == 0


@pytest.mark.parametrize("cap", (1024, 1023))
def test_support_budget_exact_required_memory_boundary(cap: int) -> None:
    with support_budget(100, 100, None, maximum_memory_bytes=cap) as budget:
        if cap == 1024:
            budget.reserve(1, 1024)
            assert budget.work_units == 1
        else:
            with pytest.raises(CoordinateEnclosureResourceError) as refusal:
                budget.reserve(1, 1024)
            assert refusal.value.limit == 1023
            assert refusal.value.requested == 1024
            assert budget.work_units == 0


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
    assert bank.row_indices.dtype == np.int64
    assert bank.row_indices.tolist() == [0, 1, 0, 1]
    assert not bank.row_indices.flags.writeable
    assert bank.row_indices.flags.c_contiguous
    assert len(bank.entries) == 2
    assert bank.entries is bank.identities.entries
    assert bank.entity_ids() == tuple(
        entity_key("authored-r1", "region", row) for row in (0, 1, 0, 1)
    )
    assert bank.roles() == (GeometrySourceEntityRole.REGION,) * 4
    assert bank.occurrence_paths() == ((),) * 4
    assert bank.entity_id(np.int64(1)) == entity_key("authored-r1", "region", 1)
    assert bank.role(1) is GeometrySourceEntityRole.REGION
    assert bank.occurrence_path(1) == ()
    leaves = jax.tree_util.tree_leaves(association)
    assert len(leaves) == 11
    assert sum(isinstance(value, jax.Array) for value in leaves) == 10
    assert sum(value is bank.row_indices for value in leaves) == 1
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
    assert len(set(bank.row_indices[:3].tolist())) == 3
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
    assert absent.identities != declared.identities
    assert absent.roles() is None
    assert declared.roles() == (None,)
    assert absent != declared


@pytest.mark.parametrize(
    "malformation",
    (
        "negative",
        "foreign",
        "boolean",
        "tuple-routes",
        "matrix-routes",
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
    entries = bank.identities.entries
    with pytest.raises(
        (TypeError, ValueError), match="GeometryAssociationSource(Bank|Identities)"
    ):
        match malformation:
            case "negative":
                replace(bank, row_indices=np.asarray((-1, 1), dtype=np.int64))
            case "foreign":
                replace(bank, row_indices=np.asarray((0, 2), dtype=np.int64))
            case "boolean":
                replace(bank, row_indices=np.asarray((False, True)))
            case "tuple-routes":
                replace(bank, row_indices=(0, 1))
            case "matrix-routes":
                replace(bank, row_indices=np.asarray(((0, 1),), dtype=np.int64))
            case "unused":
                replace(bank, row_indices=np.asarray((0, 0), dtype=np.int64))
            case "duplicate":
                GeometryAssociationSourceIdentities((entries[0], entries[0]), True)
            case "unordered":
                GeometryAssociationSourceIdentities(entries[::-1], True)
            case "mutable-path":
                GeometryAssociationSourceIdentities(
                    cast(Any, (("a", GeometrySourceEntityRole.VERTEX, []), entries[1])),
                    True,
                )
            case "untrimmed-name":
                GeometryAssociationSourceIdentities(
                    ((" a", GeometrySourceEntityRole.VERTEX, ()), entries[1]), True
                )
            case "role-type":
                GeometryAssociationSourceIdentities(
                    cast(Any, (("a", "vertex", ()), entries[1])), True
                )
            case "undeclared-role":
                GeometryAssociationSourceIdentities(entries, False)
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
    if tamper == "route":
        (route_name,) = (
            name for name, value in arrays.items() if isinstance(value, np.ndarray)
        )
        routes = np.asarray((1, 0, 1, 0), dtype=np.int64)
        routes.setflags(write=False)
        arrays[route_name] = routes
    else:
        bank_recipe = recipe["items"][recipe_field_position(recipe, "source_entities")]
        identities = bank_recipe["items"][
            recipe_field_position(bank_recipe, "identities")
        ]
        prefix = next(
            value
            for key, value in identities["payload"]["items"]
            if key["value"] == "entity_id_prefix"
        )
        prefix["value"] = "a-foreign-region:"
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
    np.testing.assert_array_equal(restored.row_indices, bank.row_indices)
    assert not restored.row_indices.flags.writeable
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
        ("face-left", "face-right", "face-left"),
        roles,
        (("assembly", "left"), ("assembly", "right"), ("assembly", "left")),
    )
    record = bank.identities.to_record()
    assert record == {
        "entity_id_prefix": "face-",
        "entity_id_suffixes": ("left", "right"),
        "role_values": ("facet",) if indexed_roles else (),
        "role_codes": (-1, 0) if indexed_roles else (-1, -1),
        "occurrence_paths": (("assembly", "left"), ("assembly", "right")),
        "occurrence_path_codes": (0, 1),
        "indexed_roles": indexed_roles,
    }
    assert bank.row_indices.tolist() == [0, 1, 0]
    assert GeometryAssociationSourceIdentities.from_record(record) == bank.identities
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(bank)
    arrays = model_recipe_array_values(bank, recipe, prefix="bank-columns")
    restored = model_from_array_recipe(recipe, arrays, prefix="bank-columns")
    assert restored == bank
    assert restored.entity_ids() == bank.entity_ids()
    assert restored.roles() == bank.roles()
    assert restored.occurrence_paths() == bank.occurrence_paths()


@pytest.mark.parametrize(
    ("identifiers", "prefix", "suffixes"),
    (
        (
            ("rev:facet:12", "rev:facet:1", "rev:vertex:3"),
            "rev:",
            ("facet:1", "facet:12", "vertex:3"),
        ),
        (("only",), "only", ("",)),
        ((), "", ()),
        (("edge", "edge:2", "edge:10"), "edge", ("", ":10", ":2")),
        (
            ("Fläche:β:2", "Fläche:β:1", "Fläche:γ:🙂"),
            "Fläche:",
            ("β:1", "β:2", "γ:🙂"),
        ),
        (("alpha", "beta", "gamma"), "", ("alpha", "beta", "gamma")),
    ),
    ids=("revision", "single", "empty", "id-equal-to-prefix", "unicode", "disjoint"),
)
def test_source_identity_record_shares_one_prefix_without_parsing_ids(
    identifiers: tuple[str, ...], prefix: str, suffixes: tuple[str, ...]
) -> None:
    bank = GeometryAssociationSourceBank.from_rows(
        identifiers, None, ((),) * len(identifiers)
    )
    identities = bank.identities
    record = identities.to_record()
    assert record["entity_id_prefix"] == prefix
    assert record["entity_id_suffixes"] == suffixes
    # The prefix is wire encoding only: decoded identities hold whole IDs.
    assert tuple(entry[0] for entry in identities.entries) == tuple(
        prefix + suffix for suffix in suffixes
    )
    assert not hasattr(identities, "entity_id_prefix")
    assert GeometryAssociationSourceIdentities.from_record(record) == identities
    register_meshing_source_artifacts()
    recipe = json.loads(
        json.dumps(model_structure_recipe(bank), ensure_ascii=True, sort_keys=True)
    )
    arrays = model_recipe_array_values(bank, recipe, prefix="bank-prefix")
    restored = model_from_array_recipe(recipe, arrays, prefix="bank-prefix")
    assert restored == bank
    assert restored.entity_ids() == identifiers


@pytest.mark.parametrize(
    ("prefix", "suffixes"),
    (("", ("",)), ("edge ", ("",)), ("", ("edge", "")), ("edge", ("", ""))),
    ids=("empty-id", "untrimmed-id", "empty-second-id", "duplicate-id"),
)
def test_source_identity_record_refuses_suffixes_without_valid_ids(
    prefix: str, suffixes: tuple[str, ...]
) -> None:
    record = {
        "entity_id_prefix": prefix,
        "entity_id_suffixes": suffixes,
        "role_values": (),
        "role_codes": (-1,) * len(suffixes),
        "occurrence_paths": ((),),
        "occurrence_path_codes": (0,) * len(suffixes),
        "indexed_roles": False,
    }
    with pytest.raises(ValueError, match="GeometryAssociationSourceIdentities"):
        GeometryAssociationSourceIdentities.from_record(record)


@pytest.mark.parametrize(
    "tamper",
    (
        "missing",
        "extra",
        "mutable",
        "extent",
        "role",
        "split",
        "code",
        "unused",
        "boolean-code",
    ),
)
def test_source_bank_column_record_refuses_malformed_wire_metadata(tamper: str) -> None:
    record = dict(_association().source_entities.identities.to_record())
    match tamper:
        case "missing":
            del record["role_codes"]
        case "extra":
            record["unknown"] = ()
        case "mutable":
            record["entity_id_suffixes"] = list(record["entity_id_suffixes"])
        case "extent":
            record["role_codes"] = ()
        case "role":
            record["role_values"] = ("foreign-role",)
        case "split":
            prefix = record["entity_id_prefix"]
            record["entity_id_prefix"] = prefix[:-1]
            record["entity_id_suffixes"] = tuple(
                prefix[-1] + suffix for suffix in record["entity_id_suffixes"]
            )
        case "code":
            record["occurrence_path_codes"] = (0, 1)
        case "unused":
            record["occurrence_paths"] = (*record["occurrence_paths"], ("unused",))
        case "boolean-code":
            record["role_codes"] = (False, False)
        case _:
            raise ValueError(
                f"source_bank.tamper must name a wire mutation; got {tamper!r}"
            )
    with pytest.raises(
        (TypeError, ValueError), match="GeometryAssociationSourceIdentities"
    ):
        GeometryAssociationSourceIdentities.from_record(record)


def test_volume_consensus_does_not_classify_exterior_boundary(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    import phydrax as phx
    from phydrax.discretization import CellMesh
    from phydrax.meshing._association import _volume_stratum_consensus
    from phydrax.meshing._association_routes import (
        boundary_mask,
        stratum_region_consensus,
    )
    from phydrax.meshing._plc_mapped_support import prepare_mapped_plc_support

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
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    domain = phx.geometry.PiecewiseLinearDomain(
        np.asarray(mesh.coordinates)[:4],
        np.asarray(
            phx.discretization.reference_cell_topology("tetrahedron").entities[2],
            dtype=np.int64,
        ),
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("body",),
        source_id="consensus-source",
    )
    support = prepare_mapped_plc_support(
        mesh,
        geometry,
        domain,
        np.zeros((4,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        maximum_support_queries=32768,
        budget=reference_budget,
    )
    for dimension in range(4):
        for labels in (
            np.zeros((4,), dtype=np.int64),
            np.asarray((0, 1, 0, 1), dtype=np.int64),
        ):
            regions = _volume_stratum_consensus(support, dimension, labels)
            expected = stratum_region_consensus(mesh, dimension, labels, 3)
            if dimension < 3:
                boundary = boundary_mask(mesh, dimension)
                expected[boundary] = -1
                assert np.any(boundary)
                assert np.all(regions[boundary] == -1)
            np.testing.assert_array_equal(regions, expected)
            assert not np.shares_memory(regions, labels)
    regions = _volume_stratum_consensus(support, 0, np.zeros((4,), dtype=np.int64))
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

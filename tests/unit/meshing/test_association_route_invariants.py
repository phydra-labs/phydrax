"""Shared association topology contracts through declared mesh and source records."""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

import equinox as eqx
import numpy as np
import pytest

from phydrax._fingerprint import tree_content_equal
from phydrax._resource_ticket import HostBatchTicket
from phydrax._strict import StrictModule
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry import Circle
from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from phydrax.meshing import (
    CellMeshingResult,
    CurveJunction,
    EntityLineage,
    EntityLineageKind,
    GeometryAssociation,
    GeometryAssociationKind,
    MeshingEntityKind,
    MeshingScope,
    MeshLineage,
    PiecewiseLinearComplex,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing._association_routes import (
    association_for_dimension,
    association_lookup,
    require_transfer_lineage,
    stratum_region_consensus,
)
from phydrax.units import METER
from tests._support.cell_meshes import unit_square
from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices
from tests._support.native_family_generation import native_family_generation
from tests._support.native_provider_cases import curve_plan, curve_spec


class AssociationCarrier(StrictModule):
    mesh: CellMesh
    associations: tuple[GeometryAssociation, ...]


def face_association(mesh: CellMesh, source: str) -> GeometryAssociation:
    ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    return GeometryAssociation(
        GeometryAssociationKind.IMPLICIT,
        source,
        "declared-revision",
        mesh.entity_set(2).entity_set_id,
        ids,
        ("implicit-zero-set",) * ids.size,
        np.zeros(ids.shape, dtype=np.float64),
    )


def test_dimension_lookup_retains_namespace_and_traversal_semantics() -> None:
    mesh = unit_square()
    first = face_association(mesh, "first")
    second = face_association(mesh, "second")
    carrier = AssociationCarrier(mesh, (first, second))
    assert association_for_dimension(carrier, 2) is first
    assert (
        association_for_dimension(carrier, 2, namespace=("second", "declared-revision"))
        is second
    )
    assert association_lookup(carrier)[2] is second
    assert 1 not in association_lookup(carrier)


def unknown_lineage(mesh: CellMesh, target_topology: str) -> MeshLineage:
    entities = mesh.entity_set(0)
    ids = np.asarray(entities.entity_ids, dtype=np.int64)
    record = EntityLineage(
        0,
        entities.entity_set_id,
        entities.entity_set_id,
        ids,
        ids,
        np.full(ids.shape, int(EntityLineageKind.UNKNOWN), dtype=np.int32),
    )
    return MeshLineage(mesh.topology_id, target_topology, (record,))


def test_lineage_binding_refuses_before_unknown_relation() -> None:
    mesh = unit_square()
    carrier = AssociationCarrier(mesh, ())
    with pytest.raises(ValueError, match="topology-binding"):
        require_transfer_lineage(
            carrier.mesh,
            unknown_lineage(mesh, "different-topology"),
            mesh,
            degrees=(0,),
            message="topology-binding",
            unknown_message="unknown-relation",
        )


def test_lineage_unknown_relation_reaches_consumer() -> None:
    mesh = unit_square()
    carrier = AssociationCarrier(mesh, ())
    with pytest.raises(ValueError, match="unknown-relation"):
        require_transfer_lineage(
            carrier.mesh,
            unknown_lineage(mesh, mesh.topology_id),
            mesh,
            degrees=(0,),
            message="topology-binding",
            unknown_message="unknown-relation",
        )


def test_planar_region_consensus_preserves_interface_refusal() -> None:
    mesh = unit_square()
    regions = np.asarray((10, 20), dtype=np.int64)
    match mesh.connectivity:
        case PolygonalConnectivity() as connectivity:
            edges = np.asarray(connectivity.edges, dtype=np.int64)
        case _:
            raise TypeError(
                f"AssociationRouteFixture.connectivity must be PolygonalConnectivity; got {type(mesh.connectivity).__name__}"
            )
    match mesh.blocks[0]:
        case CellBlock() as block:
            cells = np.asarray(block.vertices, dtype=np.int64)
            cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        case _:
            raise TypeError(
                f"AssociationRouteFixture.block must be CellBlock; got {type(mesh.blocks[0]).__name__}"
            )
    top_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    expected = []
    assert edges.ndim == 2
    assert cells.ndim == 2
    for edge_row in range(edges.shape[0]):
        edge_ids = {int(edges[edge_row, slot]) for slot in range(edges.shape[1])}
        incident_regions = {
            regions[np.flatnonzero(top_ids == cell_ids[row])[0]]
            for row in range(cells.shape[0])
            if edge_ids.issubset(int(cells[row, slot]) for slot in range(cells.shape[1]))
        }
        expected.append(
            next(iter(incident_regions)) if len(incident_regions) == 1 else -1
        )
    np.testing.assert_array_equal(stratum_region_consensus(mesh, 1, regions, 2), expected)
    top = stratum_region_consensus(mesh, 2, regions, 2)
    np.testing.assert_array_equal(top, regions)
    assert not np.shares_memory(top, regions)


def test_association_target_predicate_matches_validation_refusal() -> None:
    mesh = unit_square()
    association = face_association(mesh, "implicit-source")
    assert association.target_matches(mesh.entity_set(2))
    association.validate_target(mesh.entity_set(2))
    assert not association.target_matches(mesh.entity_set(1))
    with pytest.raises(ValueError, match=r"GeometryAssociation\.target_entity_set_id"):
        association.validate_target(mesh.entity_set(1))


def test_association_target_host_membership_does_not_compile() -> None:
    from tools.compile_counter import CompileCounter

    mesh = unit_square()
    association = face_association(mesh, "implicit-source")
    entities = mesh.entity_set(2)
    with CompileCounter() as compiler:
        assert association.target_matches(entities)
        association.validate_target(entities)
    assert compiler.count == 0


@pytest.fixture
def plc_source() -> PiecewiseLinearComplex:
    return PiecewiseLinearComplex(
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64),
        tuple(
            np.asarray(loop, dtype=np.int64)
            for loop in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        ),
        np.arange(4, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (4, 1)),
        ("material",),
    )


def test_plc_stored_fields_rebuilds_packed_loops(
    plc_source: PiecewiseLinearComplex,
) -> None:
    rebuilt = PiecewiseLinearComplex.from_stored_fields(plc_source)
    assert rebuilt is not plc_source
    assert rebuilt.complex_id == plc_source.complex_id
    assert rebuilt.boundary == plc_source.boundary
    assert rebuilt.region_ids == plc_source.region_ids
    np.testing.assert_array_equal(rebuilt.vertices, plc_source.vertices)
    np.testing.assert_array_equal(rebuilt.polygon_offsets, plc_source.polygon_offsets)
    np.testing.assert_array_equal(rebuilt.polygon_vertices, plc_source.polygon_vertices)
    np.testing.assert_array_equal(rebuilt.polygon_facets, plc_source.polygon_facets)
    np.testing.assert_array_equal(rebuilt.facet_regions, plc_source.facet_regions)
    np.testing.assert_array_equal(rebuilt.segments, plc_source.segments)


def test_plc_stored_fields_refuses_invalid_loop_partition(
    plc_source: PiecewiseLinearComplex,
) -> None:
    offsets = np.array(plc_source.polygon_offsets, copy=True)
    offsets[0] = 1
    corrupted = eqx.tree_at(lambda value: value.polygon_offsets, plc_source, offsets)
    with pytest.raises(ValueError, match="polygon_offsets"):
        PiecewiseLinearComplex.from_stored_fields(corrupted)


def test_plc_stored_fields_refuses_changed_source_identity(
    plc_source: PiecewiseLinearComplex,
) -> None:
    vertices = np.array(plc_source.vertices, copy=True)
    vertices[1, 0] = 2
    corrupted = eqx.tree_at(lambda value: value.vertices, plc_source, vertices)
    with pytest.raises(ValueError, match="complex_id"):
        PiecewiseLinearComplex.from_stored_fields(corrupted)


@pytest.mark.parametrize("resolved", (False, True), ids=("unresolved", "resolved"))
def test_implicit_named_source_accepts_absent_index_metadata(resolved: bool) -> None:
    association = GeometryAssociation(
        GeometryAssociationKind.IMPLICIT,
        "implicit-source",
        "implicit-revision",
        "face-set",
        np.asarray((42,), dtype=np.int64),
        ("implicit-zero-set",),
        np.asarray((0.0 if resolved else np.inf,), dtype=np.float64),
        resolved=np.asarray((resolved,), dtype=np.bool_),
        source_dimensions=np.asarray((-1,), dtype=np.int8),
        source_indices=np.asarray((-1,), dtype=np.int32),
    )
    assert association.source_entities.entity_ids() == ("implicit-zero-set",)
    np.testing.assert_array_equal(association.source_dimensions, (-1,))
    np.testing.assert_array_equal(association.source_indices, (-1,))
    np.testing.assert_array_equal(association.resolved, (resolved,))
    unindexed = GeometryAssociation(
        GeometryAssociationKind.IMPLICIT,
        "implicit-source",
        "implicit-revision",
        "face-set",
        np.asarray((42,), dtype=np.int64),
        ("implicit-zero-set",),
        np.asarray((0.0 if resolved else np.inf,), dtype=np.float64),
        resolved=np.asarray((resolved,), dtype=np.bool_),
    )
    assert association.association_id == unindexed.association_id


def test_implicit_unclassified_source_requires_unresolved_status() -> None:
    association = GeometryAssociation(
        GeometryAssociationKind.IMPLICIT,
        "implicit-source",
        "implicit-revision",
        "face-set",
        np.asarray((42,), dtype=np.int64),
        ("implicit-revision:unclassified",),
        np.asarray((np.inf,), dtype=np.float64),
        resolved=np.asarray((False,), dtype=np.bool_),
        source_dimensions=np.asarray((-1,), dtype=np.int8),
        source_indices=np.asarray((-1,), dtype=np.int32),
    )
    assert not association.complete
    np.testing.assert_array_equal(association.source_dimensions, (-1,))
    with pytest.raises(ValueError, match=r"GeometryAssociation\.source_entity_ids"):
        GeometryAssociation(
            GeometryAssociationKind.IMPLICIT,
            "implicit-source",
            "implicit-revision",
            "face-set",
            np.asarray((42,), dtype=np.int64),
            ("implicit-revision:unclassified",),
            np.asarray((0.0,), dtype=np.float64),
            resolved=np.asarray((True,), dtype=np.bool_),
            source_dimensions=np.asarray((-1,), dtype=np.int8),
            source_indices=np.asarray((-1,), dtype=np.int32),
        )


@pytest.mark.parametrize(
    ("dimension", "index"), ((3, 0), (1, 1)), ids=("volume-stratum", "foreign-index")
)
def test_implicit_indexed_source_refuses_malformed_resolved_strata(
    dimension: int,
    index: int,
) -> None:
    with pytest.raises(ValueError, match=r"GeometryAssociation\.implicit_strata"):
        GeometryAssociation(
            GeometryAssociationKind.IMPLICIT,
            "implicit-source",
            "implicit-revision",
            "face-set",
            np.asarray((42,), dtype=np.int64),
            ("implicit-zero-set",),
            np.asarray((0.0,), dtype=np.float64),
            source_dimensions=np.asarray((dimension,), dtype=np.int8),
            source_indices=np.asarray((index,), dtype=np.int32),
        )


@pytest.fixture(scope="module")
def unindexed_associations() -> dict[
    GeometryAssociationKind, tuple[GeometryAssociation, ...]
]:
    atlas = (
        Circle((0.0, 0.0), 1.0, feature_id="stored-curve")
        .compile(coordinate_unit=METER)
        .boundary_atlas
    )
    scope = MeshingScope(
        atlas.source_id,
        "r1",
        MeshingEntityKind.GEOMETRY,
        1,
        "arcs",
        np.arange(4, dtype=np.int64),
    )
    specification = curve_spec(
        scope,
        tuple(
            CurveJunction(f"j{i}", ((i, "end"), ((i + 1) % 4, "start"))) for i in range(4)
        ),
        size_controls=(
            UniformSizeControl(scope, 0.3, strength=SizeControlStrength.SOFT),
        ),
    )
    curve = curve_plan(atlas, specification).execute()
    part, _, _, _ = native_family_generation("mapped")
    mapped = part.carrier
    assert isinstance(mapped, CellMeshingResult)
    return {
        GeometryAssociationKind.CURVE: curve.associations,
        GeometryAssociationKind.MAPPED_REFERENCE: mapped.associations,
    }


@pytest.mark.parametrize(
    "kind",
    (GeometryAssociationKind.CURVE, GeometryAssociationKind.MAPPED_REFERENCE),
    ids=("curve", "mapped-reference"),
)
def test_unindexed_association_stored_replay_preserves_fields(
    kind: GeometryAssociationKind,
    unindexed_associations: dict[
        GeometryAssociationKind, tuple[GeometryAssociation, ...]
    ],
) -> None:
    register_meshing_source_artifacts()
    values = unindexed_associations[kind]
    assert values
    for value in values:
        assert value.association_kind is kind
        np.testing.assert_array_equal(value.source_dimensions, -1)
        np.testing.assert_array_equal(value.source_indices, -1)
        for restored in (
            GeometryAssociation.from_stored_fields(value),
            reconstruct_stored_fields(value),
        ):
            assert restored is not value
            assert restored.association_id == value.association_id
            for member in fields(value):
                assert tree_content_equal(
                    getattr(restored, member.name), getattr(value, member.name)
                ), member.name


@pytest.mark.parametrize(
    "kind",
    (GeometryAssociationKind.CURVE, GeometryAssociationKind.MAPPED_REFERENCE),
    ids=("curve", "mapped-reference"),
)
@pytest.mark.parametrize(
    ("dimension", "index", "field"),
    (
        (-2, -1, "source_dimensions"),
        (-1, -2, "source_indices"),
        (1, 0, "source_dimensions"),
    ),
    ids=("dimension-sentinel", "index-sentinel", "indexed-rows"),
)
def test_unindexed_association_stored_replay_refuses_forged_index_banks(
    kind: GeometryAssociationKind,
    dimension: int,
    index: int,
    field: str,
    unindexed_associations: dict[
        GeometryAssociationKind, tuple[GeometryAssociation, ...]
    ],
) -> None:
    register_meshing_source_artifacts()
    for value in unindexed_associations[kind]:
        changed = eqx.tree_at(
            lambda association: (
                association.source_dimensions,
                association.source_indices,
            ),
            value,
            (
                value.source_dimensions.at[0].set(np.int8(dimension)),
                value.source_indices.at[0].set(np.int32(index)),
            ),
        )
        assert changed.association_id == value.association_id
        message = (
            rf"stored_fields\.GeometryAssociation\.{field} must "
            rf"match reconstructed retained content; got changed {field}"
        )
        with pytest.raises(ValueError, match=message):
            GeometryAssociation.from_stored_fields(changed)
        with pytest.raises(ValueError, match=message):
            reconstruct_stored_fields(changed)


@pytest.mark.parametrize("indexed", (False, True))
def test_budgeted_association_replay_preserves_owner_identity(indexed: bool) -> None:
    count = 512
    if indexed:
        revision = "declared-revision"
        from phydrax.meshing import GeometrySourceEntityRole
        from phydrax.meshing._host_banks import entity_key

        roles = (GeometrySourceEntityRole.FACET,) * count
        value = GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            "source",
            revision,
            "target",
            np.arange(count, dtype=np.int64),
            (entity_key(revision, GeometrySourceEntityRole.FACET.value, 0),) * count,
            np.zeros(count),
            source_dimensions=np.full(count, 2, dtype=np.int8),
            source_indices=np.zeros(count, dtype=np.int32),
            source_entity_roles=roles,
        )
    else:
        value = GeometryAssociation(
            GeometryAssociationKind.SURFACE,
            "source",
            "declared-revision",
            "target",
            np.arange(count, dtype=np.int64),
            ("patch",) * count,
            np.zeros(count),
            source_dimensions=np.full(count, 2, dtype=np.int8),
            source_indices=np.zeros(count, dtype=np.int64),
            source_occurrence_paths=(("occurrence-" + "x" * 128,),) * count,
        )
    bank = value.source_entities
    assert len(bank.entries) == 1
    names, roles, paths = bank.row_columns()
    assert names == bank.entity_ids()
    assert roles == bank.roles()
    assert paths == bank.occurrence_paths()
    assert all(name is bank.entries[0][0] for name in names)
    assert all(path is bank.entries[0][2] for path in paths)
    with HostBatchTicket(100_000, bytes_upper=16 * 1024**2) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=16 * 1024**2
        )
        with budget.measurement():
            rebuilt = GeometryAssociation.from_stored_fields(value, budget=budget)
        assert rebuilt is not value
        assert tree_content_equal(rebuilt, value)
        assert rebuilt.association_id == value.association_id
        assert budget.temporary_bytes_upper == 0
        assert budget.retained_basis_bytes > 0
        assert budget.retained_owner_objects[id(rebuilt)] is rebuilt
        assert (
            budget.retained_owner_objects[id(rebuilt.target_global_ids)]
            is rebuilt.target_global_ids
        )


def test_budgeted_association_replay_admits_expanded_hash_buffers() -> None:
    count = 64
    value = GeometryAssociation(
        GeometryAssociationKind.SURFACE,
        "source",
        "declared-revision",
        "target",
        np.arange(count, dtype=np.int64),
        ("patch",) * count,
        np.zeros(count),
        source_dimensions=np.full(count, 2, dtype=np.int8),
        source_indices=np.zeros(count, dtype=np.int64),
        source_occurrence_paths=(("occurrence-" + "x" * 4096,),) * count,
    )
    assert len(value.source_entities.entries) == 1
    with HostBatchTicket(1_000_000, bytes_upper=4 * 1024**2) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=4 * 1024**2
        )
        with pytest.raises(CoordinateEnclosureResourceError) as error:
            with budget.measurement():
                GeometryAssociation.from_stored_fields(value, budget=budget)
        assert error.value.phase_evidence is not None
        assert error.value.requested > error.value.limit
        assert budget.temporary_bytes_upper == 0


def test_association_generic_restore_preserves_source_identity() -> None:
    value = face_association(unit_square(), "implicit-source")
    register_meshing_source_artifacts()
    restored = reconstruct_stored_fields(value)
    assert restored.association_id == value.association_id
    assert restored.source_entities.entity_ids() == value.source_entities.entity_ids()
    np.testing.assert_array_equal(restored.source_dimensions, value.source_dimensions)
    np.testing.assert_array_equal(restored.source_indices, value.source_indices)
    np.testing.assert_array_equal(restored.resolved, value.resolved)


def test_association_stored_fields_refuses_changed_numerical_identity() -> None:
    value = face_association(unit_square(), "implicit-source")
    register_meshing_source_artifacts()
    changed = eqx.tree_at(
        lambda association: association.residuals,
        value,
        value.residuals + np.float64(1),
    )
    with pytest.raises(ValueError, match="stored_fields"):
        reconstruct_stored_fields(changed)


def test_distributed_association_stored_replay_and_static_id_refusal() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(
                run_on_forced_devices,
                "tests._support.stored_replay",
                "association_round_trip",
                (),
                devices=1,
                env=gloo_rank_environment(coordinator, rank),
            )
            for rank in range(2)
        )
        records = tuple(future.result() for future in futures)
    assert tuple(record["process"] for record in records) == (0, 1)
    assert all(record["round_trip"] and record["forged_refused"] for record in records)

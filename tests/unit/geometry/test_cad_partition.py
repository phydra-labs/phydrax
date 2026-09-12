#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path

import pytest
from OCP.BOPAlgo import BOPAlgo_CellsBuilder
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._cad_revision import AssociationStatus, CADSelectionSet
from phydrax.geometry.brep import _partition as partition_module
from phydrax.geometry.brep._model import BRepEntityId
from phydrax.geometry.brep._occt import persist_occt_shape, read_occt_shape
from phydrax.geometry.brep._partition import (
    BRepPartitionHistoryError,
    BRepPartitionOperand,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionRole,
    cad_revision_from_brep_model,
    partition_brep,
)
from phydrax.meshing.providers._gmsh import GmshProvider
from phydrax.units import MILLIMETER


_COORDINATES = SpatialCoordinateContract(MILLIMETER)


def _box(x: float, length: float = 1.0):
    return BRepPrimAPI_MakeBox(gp_Pnt(x, 0.0, 0.0), length, 1.0, 1.0).Shape()


def _persist(tmp_path: Path, name: str, shape):
    return persist_occt_shape(
        shape,
        tmp_path / f"{name}.brep",
        coordinate_contract=_COORDINATES,
        linear_deflection=0.1,
        angular_deflection=0.3,
    )


def _operand(
    operand_id: str,
    model,
    role: BRepPartitionRole = BRepPartitionRole.REGION,
    *,
    targets: tuple[str, ...] = (),
):
    revision = cad_revision_from_brep_model(model)
    selection = CADSelectionSet.from_revision(
        revision,
        tuple(
            occurrence.occurrence_id
            for occurrence in revision.occurrences
            if occurrence.kind == "solid"
        ),
    )
    return BRepPartitionOperand(
        operand_id,
        model,
        role,
        selection,
        targets,
    )


def _plan(
    operands,
    precedence: tuple[str, ...],
    *,
    overwrite: bool = False,
):
    return BRepPartitionPlan(
        _COORDINATES,
        tuple(operands),
        BRepPartitionPolicy(precedence, overwrite=overwrite),
    )


def _execute(plan, destination: Path):
    return partition_brep(
        plan,
        destination=destination,
        linear_deflection=0.1,
        angular_deflection=0.3,
    )


def _solid_volumes(result) -> dict[BRepEntityId, float]:
    shape, source_format, source_digest = read_occt_shape(result.model.source_id)
    assert source_format == "brep"
    assert source_digest == result.model.source_digest
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solids = []
    while explorer.More():
        candidate = TopoDS.Solid_s(explorer.Current())
        if not any(value.IsSame(candidate) for value in solids):
            solids.append(candidate)
        explorer.Next()
    volumes = {}
    for entity_id, solid in zip(result.model.solid_ids, solids, strict=True):
        properties = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, properties)
        volumes[entity_id] = float(properties.Mass())
    return volumes


def test_touching_regions_publish_exact_interface_and_meshing_entity_sets(tmp_path):
    left = _persist(tmp_path, "left", _box(0.0))
    right = _persist(tmp_path, "right", _box(1.0))
    result = _execute(
        _plan(
            (
                _operand("left", left),
                _operand("right", right),
            ),
            ("left", "right"),
        ),
        tmp_path / "touching-result.brep",
    )

    interfaces = [
        patch for patch in result.patches if len(patch.adjacent_region_ids) == 2
    ]
    assert result.model.topology.num_solids == 2
    assert result.topological_dimension == 3
    assert len(interfaces) == 1
    assert interfaces[0].adjacent_region_ids == ("left", "right")
    assert len(interfaces[0].entity_ids) == 1
    assert set(
        entity for _, values in result.named_solid_entity_ids for entity in values
    ) == set(result.model.solid_ids)
    assert set(
        entity for _, values in result.named_face_entity_ids for entity in values
    ) == set(result.model.face_ids)
    assert result.named_region_entity_ids == result.named_solid_entity_ids
    assert result.named_patch_entity_ids == result.named_face_entity_ids
    assert result.named_edge_entity_ids == ()
    assert (
        result.report.source_region_occurrences == result.report.source_solid_occurrences
    )
    assert result.report.source_patch_occurrences == result.report.source_face_occurrences
    assert result.report.target_regions == result.model.topology.num_solids
    assert result.report.target_patches == result.model.topology.num_faces
    assert {entity.kind for region in result.regions for entity in region.entity_ids} == {
        "solid"
    }
    assert {entity.kind for patch in result.patches for entity in patch.entity_ids} == {
        "face"
    }
    assert result.region_entity_kind == "solid"
    assert result.patch_entity_kind == "face"
    assert {region.entity_kind for region in result.regions} == {"solid"}
    assert {patch.entity_kind for patch in result.patches} == {"face"}
    assert result.report.topological_dimension == 3
    assert {region.topological_dimension for region in result.regions} == {3}
    assert {patch.topological_dimension for patch in result.patches} == {3}
    provider = GmshProvider()
    left_scope = provider.entity_scope(result.model, result.region("left").entity_ids)
    interface_scope = provider.entity_scope(result.model, interfaces[0].entity_ids)
    assert left_scope.entity_dimension == 3
    assert interface_scope.entity_dimension == 2


def test_disjoint_regions_have_only_one_sided_boundary_patches(tmp_path):
    first = _persist(tmp_path, "first", _box(0.0))
    second = _persist(tmp_path, "second", _box(2.0))
    result = _execute(
        _plan(
            (_operand("first", first), _operand("second", second)),
            ("first", "second"),
        ),
        tmp_path / "disjoint-result.brep",
    )

    assert result.model.topology.num_solids == 2
    assert {patch.adjacent_region_ids for patch in result.patches} == {
        ("first",),
        ("second",),
    }
    assert all(len(indices) == 1 for indices in result.model.topology.face_solids)


def test_explicit_precedence_is_independent_of_operand_order(tmp_path):
    high = _persist(tmp_path, "high", _box(0.0, 2.0))
    low = _persist(tmp_path, "low", _box(1.0, 2.0))
    forward = _execute(
        _plan(
            (_operand("low", low), _operand("high", high)),
            ("high", "low"),
        ),
        tmp_path / "forward.brep",
    )
    reverse = _execute(
        _plan(
            (_operand("high", high), _operand("low", low)),
            ("high", "low"),
        ),
        tmp_path / "reverse.brep",
    )

    assert forward.model.source_revision == reverse.model.source_revision
    assert forward.association_graph.graph_id == reverse.association_graph.graph_id
    volumes = _solid_volumes(forward)
    assert sum(
        volumes[value] for value in forward.region("high").entity_ids
    ) == pytest.approx(2.0)
    assert sum(
        volumes[value] for value in forward.region("low").entity_ids
    ) == pytest.approx(1.0)


def test_void_subtracts_only_its_declared_target_region(tmp_path):
    left = _persist(tmp_path, "void-left", _box(0.0, 2.0))
    right = _persist(tmp_path, "void-right", _box(3.0, 2.0))
    cutting = _persist(tmp_path, "cutting", _box(1.0, 3.0))
    result = _execute(
        _plan(
            (
                _operand("right", right),
                _operand(
                    "cut",
                    cutting,
                    BRepPartitionRole.VOID,
                    targets=("left",),
                ),
                _operand("left", left),
            ),
            ("left", "right"),
        ),
        tmp_path / "targeted-void.brep",
    )

    volumes = _solid_volumes(result)
    assert sum(
        volumes[value] for value in result.region("left").entity_ids
    ) == pytest.approx(1.0)
    assert sum(
        volumes[value] for value in result.region("right").entity_ids
    ) == pytest.approx(2.0)


def test_split_and_deleted_histories_map_exact_selection_sets(tmp_path):
    region = _persist(tmp_path, "split-region", _box(0.0, 3.0))
    cutting = _persist(tmp_path, "split-cut", _box(1.0, 1.0))
    result = _execute(
        _plan(
            (
                _operand("region", region),
                _operand(
                    "void",
                    cutting,
                    BRepPartitionRole.VOID,
                    targets=("region",),
                ),
            ),
            ("region",),
        ),
        tmp_path / "split-result.brep",
    )

    graph = result.association_graph
    region_selector = graph.source_revision.select("operand:region:solid:0")
    split = graph.resolve_target(region_selector)
    mapped = graph.resolve_target_selection(
        CADSelectionSet(graph.source_revision.revision_id, (region_selector,))
    )
    void_selector = graph.source_revision.select("operand:void:solid:0")
    deleted = graph.resolve_target(void_selector)

    assert split.status is AssociationStatus.MULTIPLE
    assert split.relation == "split"
    assert len(mapped.selectors) == 2
    assert {value.kind for value in mapped.selectors} == {"solid"}
    assert deleted.status is AssociationStatus.NO_PREIMAGE
    assert deleted.relation == "deleted"
    assert (
        graph.resolve_target_selection(
            CADSelectionSet(graph.source_revision.revision_id, (void_selector,))
        ).selectors
        == ()
    )
    assert result.report.deleted_source_occurrences > 0


def test_missing_binding_history_is_unresolved_and_publishes_nothing(
    tmp_path,
    monkeypatch,
):
    class NoHistoryCellsBuilder(BOPAlgo_CellsBuilder):
        def HasHistory(self):
            return False

    region = _persist(tmp_path, "unresolved-region", _box(0.0))
    disjoint_void = _persist(tmp_path, "unresolved-void", _box(2.0))
    destination = tmp_path / "unresolved-result.brep"
    monkeypatch.setattr(
        partition_module,
        "BOPAlgo_CellsBuilder",
        NoHistoryCellsBuilder,
    )

    with pytest.raises(BRepPartitionHistoryError, match="live Boolean history"):
        _execute(
            _plan(
                (
                    _operand("region", region),
                    _operand(
                        "void",
                        disjoint_void,
                        BRepPartitionRole.VOID,
                        targets=("region",),
                    ),
                ),
                ("region",),
            ),
            destination,
        )
    assert not destination.exists()
    assert not tuple(tmp_path.glob(f".{destination.name}.partition-*"))


def test_staging_failure_does_not_clobber_existing_destination(tmp_path, monkeypatch):
    region = _persist(tmp_path, "failure-region", _box(0.0))
    disjoint_void = _persist(tmp_path, "failure-void", _box(2.0))
    destination = tmp_path / "existing.brep"
    destination.write_bytes(b"existing-artifact")

    def fail_persistence(*args, **kwargs):
        raise RuntimeError("staged persistence failed")

    monkeypatch.setattr(
        partition_module,
        "persist_occt_shape",
        fail_persistence,
    )
    with pytest.raises(RuntimeError, match="staged persistence failed"):
        _execute(
            _plan(
                (
                    _operand("region", region),
                    _operand(
                        "void",
                        disjoint_void,
                        BRepPartitionRole.VOID,
                        targets=("region",),
                    ),
                ),
                ("region",),
                overwrite=True,
            ),
            destination,
        )
    assert destination.read_bytes() == b"existing-artifact"
    assert not tuple(tmp_path.glob(f".{destination.name}.partition-*"))

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from pathlib import Path

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._cad_revision import CADOccurrence, CADRevision
from phydrax.geometry.brep._boolean import BRepBooleanFailure
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_cylinder,
    brep_sphere,
    BRepTessellationPolicy,
)
from phydrax.geometry.brep._intersection_curve import IntersectionCurve
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._partition import (
    BRepPartitionOperand,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionResult,
    BRepPartitionRole,
    partition_brep,
)
from phydrax.geometry.brep._query import prepare_brep_query
from phydrax.interchange._cad import CadInterchangeError
from phydrax.interchange._cad_archive import load_brep_archive, save_brep_archive
from phydrax.units import MILLIMETER


_COORDINATES = SpatialCoordinateContract(MILLIMETER)
_EXACT_SOURCE = BRepTessellationPolicy(realize=False)


def _box(x: float, length: float = 1.0) -> BRepModel:
    return brep_box(
        (x, 0.0, 0.0), (x + length, 1.0, 1.0), coordinate_contract=_COORDINATES
    )


def _operand(name: str, model: BRepModel) -> BRepPartitionOperand:
    return BRepPartitionOperand(name, model, BRepPartitionRole.REGION)


def _plan(
    operands: tuple[BRepPartitionOperand, ...],
    precedence: tuple[str, ...],
    *,
    overwrite: bool = False,
) -> BRepPartitionPlan:
    return BRepPartitionPlan(
        _COORDINATES, operands, BRepPartitionPolicy(precedence, overwrite=overwrite)
    )


def _region_volume(
    result: BRepPartitionResult, name: str, model: BRepModel | None = None
) -> float:
    """Region volume from the native exact query of ``model`` (default: the result)."""
    measured = result.model if model is None else model
    volumes = np.asarray(prepare_brep_query(measured).measures.solid_volumes)
    return float(sum(volumes[entity.index] for entity in result.region(name).entity_ids))


def _assert_shared_interface(result: BRepPartitionResult, name: str) -> None:
    """Every interface face bounds exactly its two regions with opposite orientation."""
    topology = result.model.topology
    faces = result.patch(name).entity_ids
    assert faces
    for entity in faces:
        solids = topology.face_solids[entity.index]
        assert len(solids) == 2
        signs = tuple(
            topology.solid_face_orientations[solid][
                topology.solid_faces[solid].index(entity.index)
            ]
            for solid in solids
        )
        assert signs[0] == -signs[1]


def test_touching_regions_publish_exact_shared_interface(tmp_path: Path) -> None:
    result = partition_brep(
        _plan(
            (_operand("left", _box(0.0)), _operand("right", _box(1.0))), ("left", "right")
        ),
        destination=tmp_path / "touching.phx",
    )
    interface = result.patch("interface:left:right")
    assert interface.adjacent_region_ids == ("left", "right")
    assert len(interface.entity_ids) == 1
    interface_face = interface.entity_ids[0].index
    assert result.model.topology.face_solids[interface_face] == (0, 1)
    signs = tuple(
        result.model.topology.solid_face_orientations[solid][
            result.model.topology.solid_faces[solid].index(interface_face)
        ]
        for solid in (0, 1)
    )
    assert signs[0] == -signs[1]
    assert _region_volume(result, "left") == pytest.approx(1.0)
    assert _region_volume(result, "right") == pytest.approx(1.0)


def test_disjoint_regions_have_only_one_sided_boundaries(tmp_path: Path) -> None:
    result = partition_brep(
        _plan(
            (_operand("first", _box(0.0)), _operand("second", _box(2.0))),
            ("first", "second"),
        ),
        destination=tmp_path / "disjoint.phx",
    )
    assert all(len(patch.adjacent_region_ids) == 1 for patch in result.patches)
    assert result.model.topology.num_solids == 2
    assert _region_volume(result, "first") == pytest.approx(1.0)
    assert _region_volume(result, "second") == pytest.approx(1.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_precedence_is_independent_of_operand_order(
    tmp_path: Path, reverse: bool
) -> None:
    high, low = _operand("high", _box(0.0, 2.0)), _operand("low", _box(1.0, 2.0))
    operands = (low, high) if reverse else (high, low)
    result = partition_brep(
        _plan(operands, ("high", "low")), destination=tmp_path / "precedence.phx"
    )
    assert _region_volume(result, "high") == pytest.approx(2.0)
    assert _region_volume(result, "low") == pytest.approx(1.0)
    assert result.patch("interface:high:low").adjacent_region_ids == ("high", "low")


def test_void_empties_material_owned_by_its_target_without_refill(tmp_path: Path) -> None:
    # left [0,2] wins [1,2] over right [1,3]; the void empties that owned cell
    # and the lower-precedence right region does not reclaim it.
    void = BRepPartitionOperand(
        "void", _box(1.0), BRepPartitionRole.VOID, target_region_ids=("left",)
    )
    plan = _plan(
        (_operand("left", _box(0.0, 2.0)), _operand("right", _box(1.0, 2.0)), void),
        ("left", "right"),
    )
    result = partition_brep(plan, destination=tmp_path / "void.phx")
    assert _region_volume(result, "left") == pytest.approx(1.0)
    assert _region_volume(result, "right") == pytest.approx(1.0)
    assert all(len(patch.adjacent_region_ids) == 1 for patch in result.patches)


def test_split_and_deleted_histories_are_exhaustive(tmp_path: Path) -> None:
    region = _operand("body", _box(0.0, 3.0))
    void = BRepPartitionOperand(
        "void", _box(1.0), BRepPartitionRole.VOID, target_region_ids=("body",)
    )
    result = partition_brep(
        _plan((region, void), ("body",)), destination=tmp_path / "split.phx"
    )
    assert len(result.region("body").entity_ids) == 2
    graph = result.association_graph
    descendants = {
        edge.target_occurrence_id
        for edge in graph.transaction.correspondences
        if edge.source_occurrence_id == "operand:0/solid:0"
    }
    assert descendants == {"solid:0", "solid:1"}
    assert graph.transaction.coverage.source_exhaustive
    assert graph.transaction.coverage.target_exhaustive
    assert result.report.deleted_source_occurrences > 0


def test_native_archive_partition_roundtrip_preserves_revision(tmp_path: Path) -> None:
    path = tmp_path / "partition.phx"
    result = partition_brep(
        _plan((_operand("body", _box(0.0)),), ("body",)), destination=path
    )
    restored = load_brep_archive(path)
    assert restored.model_id == result.model.model_id
    assert restored.source_revision == result.revision.revision_id
    assert _region_volume(result, "body") == pytest.approx(1.0)


def test_native_external_brep_partition_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "partition.brep"
    result = partition_brep(
        _plan((_operand("body", _box(0.0)),), ("body",)), destination=path
    )
    assert result.model.source_id == str(path)
    assert _region_volume(result, "body") == pytest.approx(1.0)
    assert result.association_graph.target_revision == result.revision


def test_failure_does_not_clobber_existing_destination(tmp_path: Path) -> None:
    path = tmp_path / "unchanged.phx"
    path.write_bytes(b"accepted artifact")
    region = _operand("body", _box(0.0))
    void = BRepPartitionOperand(
        "erase", _box(0.0), BRepPartitionRole.VOID, target_region_ids=("body",)
    )
    with pytest.raises(BRepBooleanFailure, match="no surviving"):
        partition_brep(_plan((region, void), ("body",), overwrite=True), destination=path)
    assert path.read_bytes() == b"accepted artifact"
    assert not tuple(tmp_path.glob(".unchanged.phx.partition-*"))


def test_existing_destination_requires_explicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "protected.phx"
    path.write_bytes(b"accepted artifact")
    with pytest.raises(FileExistsError):
        partition_brep(_plan((_operand("body", _box(0.0)),), ("body",)), destination=path)
    assert path.read_bytes() == b"accepted artifact"


def test_revision_children_rejects_forged_parent_selector() -> None:
    parent = CADOccurrence("revision", "solid", "entity", "solid", ("solid",))
    child = CADOccurrence(
        "revision", "face", "face-entity", "face", ("solid", "face"), "solid"
    )
    revision = CADRevision("revision", "source", (parent, child), "provenance")
    from dataclasses import replace

    forged = replace(revision.select("solid"), entity_id="forged")
    with pytest.raises(ValueError):
        revision.children(forged)


def test_partition_cell_budget_fails_before_publication(tmp_path: Path) -> None:
    destination = tmp_path / "limited.phx"
    operands = (_operand("first", _box(0.0, 2.0)), _operand("second", _box(1.0, 2.0)))
    plan = BRepPartitionPlan(
        _COORDINATES, operands, BRepPartitionPolicy(("first", "second"), maximum_cells=1)
    )
    with pytest.raises(BRepBooleanFailure, match="cell budget exhausted"):
        partition_brep(plan, destination=destination)
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".limited.phx.partition-*"))


def test_operand_reordering_preserves_scientific_partition_identity(
    tmp_path: Path,
) -> None:
    high, low = _operand("high", _box(0.0, 2.0)), _operand("low", _box(1.0, 2.0))
    first = partition_brep(
        _plan((high, low), ("high", "low")), destination=tmp_path / "first.phx"
    )
    second = partition_brep(
        _plan((low, high), ("high", "low")), destination=tmp_path / "second.phx"
    )
    assert first.model.model_id == second.model.model_id
    assert first.association_graph.graph_id == second.association_graph.graph_id


# Curved scenarios. Volumes are independent closed forms: unit spheres at
# distance one share a lens of 5*pi/12; a radius-r sphere whose center lies d
# below a plane keeps 4*pi*r^3/3 - pi*a^2*(3r - a)/3 below it, a = r - d.
def _sphere(radius: float, center: tuple[float, float, float]) -> BRepModel:
    return brep_sphere(
        radius,
        center=center,
        coordinate_contract=_COORDINATES,
        tessellation=_EXACT_SOURCE,
    )


def test_overlapping_sphere_precedence_publishes_exact_curved_interface(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spheres.phx"
    result = partition_brep(
        _plan(
            (
                _operand("low", _sphere(1.0, (0.0, 1.0, 0.0))),
                _operand("high", _sphere(1.0, (0.0, 0.0, 0.0))),
            ),
            ("high", "low"),
        ),
        destination=path,
    )
    geometry = result.model.geometry
    assert geometry is not None
    assert any(isinstance(curve, IntersectionCurve) for curve in geometry.curves)
    assert result.model.topology.num_solids == 2
    assert result.patch("interface:high:low").adjacent_region_ids == ("high", "low")
    _assert_shared_interface(result, "interface:high:low")
    coverage = result.association_graph.transaction.coverage
    assert coverage.source_exhaustive and coverage.target_exhaustive
    restored = load_brep_archive(path)
    assert restored.model_id == result.model.model_id
    assert result.model.chart_restriction_ids
    assert restored.chart_restriction_ids == result.model.chart_restriction_ids
    np.testing.assert_array_equal(
        restored.mesh_chart_restriction_vertices,
        result.model.mesh_chart_restriction_vertices,
    )
    np.testing.assert_array_equal(
        restored.mesh_chart_restriction_parameters,
        result.model.mesh_chart_restriction_parameters,
    )
    assert restored.source_revision == result.revision.revision_id
    assert _region_volume(result, "high", restored) == pytest.approx(
        4.0 * math.pi / 3.0, rel=1e-6
    )
    assert _region_volume(result, "low", restored) == pytest.approx(
        11.0 * math.pi / 12.0, rel=1e-6
    )


def test_curved_intersection_branch_refuses_inexact_brep_text(tmp_path: Path) -> None:
    path = tmp_path / "spheres.brep"
    path.write_bytes(b"accepted artifact")
    plan = BRepPartitionPlan(
        _COORDINATES,
        (
            _operand("low", _sphere(1.0, (0.0, 1.0, 0.0))),
            _operand("high", _sphere(1.0, (0.0, 0.0, 0.0))),
        ),
        BRepPartitionPolicy(("high", "low"), overwrite=True),
    )
    with pytest.raises(CadInterchangeError, match="approximation"):
        partition_brep(plan, destination=path)
    assert path.read_bytes() == b"accepted artifact"
    assert not tuple(tmp_path.glob(".spheres.brep.partition-*"))


def test_contained_sphere_region_leaves_exact_cavity_in_cylinder(tmp_path: Path) -> None:
    path = tmp_path / "contained.brep"
    jacket = brep_cylinder(
        2.0,
        4.0,
        base_center=(0.0, 0.0, -2.0),
        coordinate_contract=_COORDINATES,
        tessellation=_EXACT_SOURCE,
    )
    result = partition_brep(
        _plan(
            (_operand("jacket", jacket), _operand("core", _sphere(1.0, (0.0, 0.0, 0.0)))),
            ("core", "jacket"),
        ),
        destination=path,
    )
    assert result.model.source_id == str(path)
    assert result.model.topology.num_solids == 2
    _assert_shared_interface(result, "interface:core:jacket")
    (jacket_solid,) = result.region("jacket").entity_ids
    geometry = result.model.geometry
    assert geometry is not None
    assert len(geometry.solid_shells[jacket_solid.index]) == 2
    assert _region_volume(result, "core") == pytest.approx(4.0 * math.pi / 3.0, rel=1e-6)
    assert _region_volume(result, "jacket") == pytest.approx(
        16.0 * math.pi - 4.0 * math.pi / 3.0, rel=1e-6
    )
    coverage = result.association_graph.transaction.coverage
    assert coverage.source_exhaustive and coverage.target_exhaustive
    archive = save_brep_archive(result.model, tmp_path / "contained.phx")
    restored = load_brep_archive(archive.path)
    assert restored.model_id == result.model.model_id
    assert restored.geometry is not None
    assert restored.geometry.geometry_id == geometry.geometry_id
    assert _region_volume(result, "core", restored) == pytest.approx(
        4.0 * math.pi / 3.0, rel=1e-6
    )
    assert _region_volume(result, "jacket", restored) == pytest.approx(
        16.0 * math.pi - 4.0 * math.pi / 3.0, rel=1e-6
    )


def test_curved_void_removes_only_its_target_material(tmp_path: Path) -> None:
    path = tmp_path / "void.phx"
    body = brep_cylinder(
        1.0,
        2.0,
        base_center=(0.0, 0.0, -1.0),
        coordinate_contract=_COORDINATES,
        tessellation=_EXACT_SOURCE,
    )
    void = BRepPartitionOperand(
        "void",
        _sphere(0.5, (0.0, 0.0, 0.75)),
        BRepPartitionRole.VOID,
        target_region_ids=("body",),
    )
    result = partition_brep(
        _plan((_operand("body", body), void), ("body",)), destination=path
    )
    assert result.model.topology.num_solids == 1
    assert all(len(patch.adjacent_region_ids) == 1 for patch in result.patches)
    restored = load_brep_archive(path)
    assert result.model.mesh_faces.shape[0] > 0
    assert np.all(np.asarray(result.model.tessellation_deviation_bounds) <= 1.0e-3)
    assert restored.model_id == result.model.model_id
    assert restored.chart_restriction_ids == result.model.chart_restriction_ids
    assert _region_volume(result, "body", restored) == pytest.approx(
        119.0 * math.pi / 64.0, rel=1e-6
    )
    graph = result.association_graph
    assert graph.transaction.coverage.source_exhaustive
    assert graph.transaction.coverage.target_exhaustive
    descendants = {
        edge.target_occurrence_id
        for edge in graph.transaction.correspondences
        if edge.source_occurrence_id == "operand:1/solid:0"
    }
    assert not descendants
    assert result.report.deleted_source_occurrences > 0

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax._strict import StrictModule
from phydrax.geometry import PlanarEmbedding
from phydrax.geometry._cad_revision import AssociationGraph, OccurrenceCorrespondence
from phydrax.geometry.brep import (
    brep_box,
    BRepPartitionPolicy,
    BRepPartitionResult,
    BRepPartitionRole,
    BRepTessellationPolicy,
    partition_planar,
    PlanarPartitionOperand,
    PlanarPartitionPlan,
)
from phydrax.geometry.brep._sewing import (
    BRepSewingEdgeImage,
    BRepSewingLineage,
    sew_brep,
)
from phydrax.geometry.design import (
    name_partition_entities,
    RecipeEntityName,
    RecipeEntityNames,
    RecipeNamingError,
    transport_recipe_names,
)
from phydrax.geometry.process import (
    lower_process_stack,
    ProcessStack,
    StackRegion,
    ZInterval,
)
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.units import MILLIMETER


_COORDINATES = SpatialCoordinateContract(MILLIMETER)
_EMBEDDING = PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))


def _rectangle(width: float) -> PlanarMeshRegion:
    return PlanarMeshRegion(
        np.asarray(((0, 0), (width, 0), (width, 2), (0, 2)), dtype=np.float64),
        ((0, 1, 2, 3),),
        feature_id="authored-rectangle",
    )


def _planar(width: float, destination: Path) -> BRepPartitionResult:
    plan = PlanarPartitionPlan(
        _COORDINATES,
        _EMBEDDING,
        (PlanarPartitionOperand("metal", _rectangle(width), BRepPartitionRole.REGION),),
        BRepPartitionPolicy(("metal",)),
    )
    return partition_planar(plan, destination=destination)


def _sources() -> dict[str, RecipeEntityName]:
    # These bindings are authored against the rectangular source loop, not
    # recovered from target index order or the implementation's returned names.
    return {
        "operand:metal:face:0": RecipeEntityName("face", "metal", "plate", ("layout",)),
        "operand:metal:face:0/edge:0": RecipeEntityName(
            "edge", "metal", "south", ("layout",)
        ),
        "operand:metal:face:0/edge:1": RecipeEntityName(
            "edge", "metal", "east", ("layout",)
        ),
        "operand:metal:face:0/edge:2": RecipeEntityName(
            "edge", "metal", "north", ("layout",)
        ),
        "operand:metal:face:0/edge:3": RecipeEntityName(
            "edge", "metal", "west", ("layout",)
        ),
    }


def test_names_have_value_identity_without_geometry_or_revision() -> None:
    first = RecipeEntityName("face", "metal", "plate", ("layout", "partition"))
    second = RecipeEntityName("face", "metal", "plate", ("layout", "partition"))
    assert isinstance(first, StrictModule)
    assert first == second
    assert hash(first) == hash(second)
    assert first.key == ("face", "metal", "plate", ("layout", "partition"))
    assert not hasattr(first, "model_id")
    assert not hasattr(first, "source_revision")
    assert first != RecipeEntityName("face", "metal", "other", ("layout", "partition"))


def test_planar_recipe_names_equal_independent_expectations_across_edits(
    tmp_path: Path,
) -> None:
    first = _planar(1, tmp_path / "first.phx")
    edited = _planar(3, tmp_path / "edited.phx")
    original = name_partition_entities(first, _sources())
    changed = name_partition_entities(edited, _sources())
    expected_face = RecipeEntityName("face", "metal", "plate", ("layout", "partition"))
    expected_edges = {
        RecipeEntityName("edge", "metal", "south", ("layout", "partition")),
        RecipeEntityName("edge", "metal", "east", ("layout", "partition")),
        RecipeEntityName("edge", "metal", "north", ("layout", "partition")),
        RecipeEntityName("edge", "metal", "west", ("layout", "partition")),
    }
    assert first.model.model_id != edited.model.model_id
    assert first.model.source_revision != edited.model.source_revision
    assert original.face_names == changed.face_names == (expected_face,)
    assert original.edge_names is not None and changed.edge_names is not None
    assert set(original.edge_names) == set(changed.edge_names) == expected_edges
    for bindings, model in ((original, first.model), (changed, edited.model)):
        assert bindings.model_id == model.model_id
        assert bindings.resolve(expected_face) == model.face_ids[0]
        for entity in (*model.face_ids, *model.edge_ids):
            assert bindings.resolve(bindings.name(entity)) == entity
    with pytest.raises(KeyError, match="this model"):
        changed.name(first.model.face_ids[0])


@pytest.mark.parametrize("relation", ["split", "merge"])
def test_partition_names_refuse_split_and_merge(tmp_path: Path, relation: str) -> None:
    result = _planar(1, tmp_path / "partition.phx")
    graph = result.association_graph
    pairs = graph.transaction.correspondences
    edge_pairs = tuple(pair for pair in pairs if "/edge:" in pair.source_occurrence_id)
    first, second = edge_pairs[:2]
    if relation == "split":
        replacement = OccurrenceCorrespondence(
            first.source_occurrence_id, second.target_occurrence_id, first.evidence_id
        )
    else:
        replacement = OccurrenceCorrespondence(
            second.source_occurrence_id, first.target_occurrence_id, first.evidence_id
        )
    transaction = replace(
        graph.transaction,
        correspondences=tuple(pair for pair in pairs if pair != second) + (replacement,),
    )
    changed = replace(
        result,
        association_graph=AssociationGraph(
            graph.source_revision, graph.target_revision, transaction
        ),
    )
    with pytest.raises(RecipeNamingError, match=relation):
        name_partition_entities(changed, _sources())


def test_partition_names_refuse_incomplete_lineage_and_authored_bindings(
    tmp_path: Path,
) -> None:
    result = _planar(1, tmp_path / "partition.phx")
    graph = result.association_graph
    transaction = replace(
        graph.transaction,
        coverage=replace(graph.transaction.coverage, target_exhaustive=False),
    )
    changed = replace(
        result,
        association_graph=AssociationGraph(
            graph.source_revision, graph.target_revision, transaction
        ),
    )
    with pytest.raises(RecipeNamingError, match="exhaustive"):
        name_partition_entities(changed, _sources())
    incomplete = _sources()
    del incomplete["operand:metal:face:0/edge:0"]
    with pytest.raises(RecipeNamingError, match="source face/edge"):
        name_partition_entities(result, incomplete)
    ambiguous = _sources()
    ambiguous["operand:metal:face:0/edge:0"] = ambiguous["operand:metal:face:0/edge:1"]
    with pytest.raises(RecipeNamingError, match="ambiguous identity"):
        name_partition_entities(result, ambiguous)


def test_real_partition_split_refuses_recipe_names(tmp_path: Path) -> None:
    # A vertical strip removes the middle of a plate: the source face splits.
    void = PlanarMeshRegion(
        np.asarray(((1, -1), (2, -1), (2, 3), (1, 3)), dtype=np.float64),
        ((0, 1, 2, 3),),
        feature_id="opening",
    )
    plan = PlanarPartitionPlan(
        _COORDINATES,
        _EMBEDDING,
        (
            PlanarPartitionOperand("metal", _rectangle(3), BRepPartitionRole.REGION),
            PlanarPartitionOperand(
                "opening", void, BRepPartitionRole.VOID, target_region_ids=("metal",)
            ),
        ),
        BRepPartitionPolicy(("metal",)),
    )
    result = partition_planar(plan, destination=tmp_path / "split.phx")
    with pytest.raises(RecipeNamingError, match="split"):
        name_partition_entities(result, _sources())


def test_process_recipe_faces_bind_and_uncertified_edges_refuse(tmp_path: Path) -> None:
    stack = ProcessStack(
        _COORDINATES,
        (StackRegion("metal", _rectangle(1), ZInterval(0, 1, MILLIMETER), 0),),
    )
    result = lower_process_stack(stack, tmp_path / "stack.brep")
    # Authored extrusion bindings are explicitly supplied, never manufactured
    # from a changed model's revision or geometry fingerprint.
    sources = {
        f"operand:0/solid:0/face:{index}": RecipeEntityName(
            "face", "metal", label, ("extrude",)
        )
        for index, label in enumerate(("bottom", "top", "south", "east", "north", "west"))
    }
    names = result.recipe_names(sources)
    assert names.face_names is not None
    assert names.edge_names is None
    assert {name.source_entity for name in names.face_names} == {
        "bottom",
        "top",
        "south",
        "east",
        "north",
        "west",
    }
    for entity in result.model.face_ids:
        assert names.resolve(names.name(entity)) == entity
    with pytest.raises(RecipeNamingError, match="edge lineage is unsupported"):
        names.name(result.model.edge_ids[0])


def test_identity_sewing_names_equal_expected_across_parameter_edits() -> None:
    results = []
    for width in (1, 3):
        model = brep_box(
            (0, 0, 0),
            (width, 2, 1),
            coordinate_contract=_COORDINATES,
            tessellation=BRepTessellationPolicy(realize=False),
        )
        geometry = model.geometry
        assert geometry is not None
        sewn = sew_brep(geometry, model.orientation, model.topology.solid_faces)
        faces = tuple(
            RecipeEntityName("face", "metal", label, ("extrude",))
            for label in ("bottom", "top", "south", "east", "north", "west")
        )
        edges = tuple(
            RecipeEntityName("edge", "metal", label, ("extrude",))
            for label in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l")
        )
        source = RecipeEntityNames(model, faces, edges)
        names = transport_recipe_names(source, model, sewn.lineage)
        assert names.face_names is not None and names.edge_names is not None
        assert names.face_names == tuple(
            RecipeEntityName("face", "metal", label, ("extrude", "sewing"))
            for label in ("bottom", "top", "south", "east", "north", "west")
        )
        assert names.edge_names == tuple(
            RecipeEntityName("edge", "metal", label, ("extrude", "sewing"))
            for label in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l")
        )
        results.append(names)
        with pytest.raises(RecipeNamingError, match="exhaustive sewing edge"):
            transport_recipe_names(
                source,
                model,
                BRepSewingLineage(
                    sewn.lineage.vertex_sources,
                    sewn.lineage.edge_images[:-1],
                    sewn.lineage.coedge_sources,
                    sewn.lineage.pair_gap_bounds,
                    sewn.lineage.edge_face_gap_coedges,
                ),
            )
        with pytest.raises(RecipeNamingError, match="split"):
            transport_recipe_names(
                source,
                model,
                BRepSewingLineage(
                    sewn.lineage.vertex_sources,
                    (
                        *sewn.lineage.edge_images,
                        BRepSewingEdgeImage(0, 1, Fraction(1), Fraction(0), Fraction(0)),
                    ),
                    sewn.lineage.coedge_sources,
                    sewn.lineage.pair_gap_bounds,
                    sewn.lineage.edge_face_gap_coedges,
                ),
            )
        merged = tuple(
            image for image in sewn.lineage.edge_images if image.source_edge != 1
        )
        with pytest.raises(RecipeNamingError, match="merge"):
            transport_recipe_names(
                source,
                model,
                BRepSewingLineage(
                    sewn.lineage.vertex_sources,
                    (
                        *merged,
                        BRepSewingEdgeImage(1, 0, Fraction(1), Fraction(0), Fraction(0)),
                    ),
                    sewn.lineage.coedge_sources,
                    sewn.lineage.pair_gap_bounds,
                    sewn.lineage.edge_face_gap_coedges,
                ),
            )
        with pytest.raises(RecipeNamingError, match="coedge lineage"):
            transport_recipe_names(
                source,
                model,
                BRepSewingLineage(
                    sewn.lineage.vertex_sources,
                    sewn.lineage.edge_images,
                    sewn.lineage.coedge_sources[:-1],
                    sewn.lineage.pair_gap_bounds,
                    sewn.lineage.edge_face_gap_coedges,
                ),
            )
    assert results[0].face_names == results[1].face_names
    assert results[0].edge_names == results[1].edge_names

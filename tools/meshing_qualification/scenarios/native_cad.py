from __future__ import annotations

from typing import Any

import numpy as np

import phydrax as phx


def _contract() -> phx.SpatialCoordinateContract:
    return phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _planar_region(name: str, x0: float, x1: float) -> phx.geometry.PlanarMeshRegion:
    return phx.geometry.PlanarMeshRegion(
        np.asarray(((x0, 0.0), (x1, 0.0), (x1, 1.0), (x0, 1.0))),
        ((0, 1, 2, 3),),
        feature_id=name,
    )


def _planar_partition() -> Any:
    contract = _contract()
    embedding = phx.geometry.PlanarEmbedding(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    operands = (
        phx.geometry.PlanarPartitionOperand(
            "left",
            _planar_region("left", 0.0, 1.0),
            phx.geometry.BRepPartitionRole.REGION,
        ),
        phx.geometry.PlanarPartitionOperand(
            "right",
            _planar_region("right", 1.0, 2.0),
            phx.geometry.BRepPartitionRole.REGION,
        ),
    )
    plan = phx.geometry.PlanarPartitionPlan(
        contract,
        embedding,
        operands,
        phx.geometry.BRepPartitionPolicy(region_precedence=("right", "left")),
    )
    return embedding, phx.geometry.partition_planar(plan)


def _three_dimensional_partition() -> Any:
    contract = _contract()
    operands = (
        phx.geometry.BRepPartitionOperand(
            "left",
            phx.geometry.brep.brep_box(
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                coordinate_contract=contract,
                source_id="left",
            ),
            phx.geometry.BRepPartitionRole.REGION,
        ),
        phx.geometry.BRepPartitionOperand(
            "right",
            phx.geometry.brep.brep_box(
                (1.0, 0.0, 0.0),
                (2.0, 1.0, 1.0),
                coordinate_contract=contract,
                source_id="right",
            ),
            phx.geometry.BRepPartitionRole.REGION,
        ),
    )
    plan = phx.geometry.BRepPartitionPlan(
        contract,
        operands,
        phx.geometry.BRepPartitionPolicy(region_precedence=("right", "left")),
    )
    return phx.geometry.partition_brep(plan)


def qualify_cad_partition() -> dict[str, object]:
    result = _three_dimensional_partition()
    if result.topological_dimension != 3 or not result.named_solid_entity_ids:
        raise RuntimeError("CAD partition did not produce exact solid regions.")
    if not result.association_graph.graph_id or not result.revision.revision_id:
        raise RuntimeError("CAD partition omitted revision association evidence.")
    return {
        "coordinate_contract": result.model.coordinate_contract.spatial_id,
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "solid_regions": {name: len(ids) for name, ids in result.named_solid_entity_ids},
        "face_patches": {name: len(ids) for name, ids in result.named_face_entity_ids},
    }


def qualify_planar_semantic() -> dict[str, object]:
    embedding, result = _planar_partition()
    if (
        result.topological_dimension != 2
        or result.model.topology.num_solids != 0
        or not result.named_region_entity_ids
        or not result.named_patch_entity_ids
    ):
        raise RuntimeError("Planar partition did not preserve two-dimensional semantics.")
    return {
        "embedding": embedding.embedding_id,
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "region_entity_kind": result.region_entity_kind,
        "patch_entity_kind": result.patch_entity_kind,
        "regions": {name: len(ids) for name, ids in result.named_region_entity_ids},
        "patches": {name: len(ids) for name, ids in result.named_patch_entity_ids},
    }

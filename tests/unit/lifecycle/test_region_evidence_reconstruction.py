# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Region source archives require retained result mesh and inventory bindings."""

from copy import copy
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import tree_content_equal
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    reconstruct_meshing_source,
    register_meshing_source_artifacts,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    CellMeshingResult,
    RegionBoundaryEvidence,
    RegionMeshingEvidence,
)
from tests._support.cell_meshes import unit_square
from tests._support.image_sources import _execute, _source
from tests._support.meshing_fixtures import mesh_scope


@pytest.fixture(scope="module", params=("volume", "boundary"))
def region_owner(request: pytest.FixtureRequest) -> CellMeshingResult:
    register_meshing_source_artifacts()
    if request.param == "volume":
        values = np.ones((2, 2, 2), dtype=np.int16)
        values[1] = 2
        return _execute(_source(values))
    mesh = unit_square()
    scope = mesh_scope(mesh, 2, mesh.entity_set(2).entity_ids)
    label = phx.meshing.MeshLabel("source-region-boundary", scope)
    patch = phx.meshing.MeshPatch(
        "source-region-wall", scope, source_adjacent_region_ids=("fluid", None)
    )
    source_scope = phx.meshing.MeshingScope(
        "source",
        "revision",
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        "source-regions",
        np.asarray((1,), dtype=np.int64),
    )
    evidence = RegionBoundaryEvidence(
        mesh,
        source_scope,
        "fluid",
        "boundary-control",
        "water",
        phx.meshing.RegionRole.FLUID,
        label,
        ((patch.patch_id, 1),),
    )
    return phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        labels=(label,),
        patches=(patch,),
        region_boundary_evidence=(evidence,),
    )


def _evidence(owner: CellMeshingResult) -> RegionMeshingEvidence | RegionBoundaryEvidence:
    if owner.region_evidence is not None:
        return owner.region_evidence
    return owner.region_boundary_evidence[0]


def test_region_evidence_archive_round_trip_binds_retained_result(
    region_owner: CellMeshingResult,
    tmp_path: Path,
) -> None:
    evidence = _evidence(region_owner)
    path = tmp_path / "region-source.zip"
    receipt = write_meshing_source_closure(path, (evidence, region_owner))
    restored, owner = read_meshing_source_closure(
        path,
        expected_content_id=receipt.content_id,
    )
    assert tree_content_equal(restored, evidence)
    assert owner.result_id == region_owner.result_id
    rebound = reconstruct_meshing_source(restored, results=(owner,))
    assert tree_content_equal(rebound, evidence)
    match restored:
        case RegionMeshingEvidence():
            restored.require_current(
                owner.mesh, owner.zones, owner.patches, geometry=owner.geometry
            )
        case RegionBoundaryEvidence():
            restored.require_current(owner.mesh, owner.labels, owner.patches)


@pytest.mark.parametrize("foreign", ("mesh", "topology", "inventory", "identity"))
def test_region_evidence_refuses_foreign_result_binding(
    region_owner: CellMeshingResult,
    foreign: str,
) -> None:
    evidence = copy(_evidence(region_owner))
    match foreign:
        case "mesh":
            object.__setattr__(evidence, "target_mesh_id", "foreign-mesh")
        case "topology":
            object.__setattr__(evidence, "target_topology_id", "foreign-topology")
        case "identity":
            object.__setattr__(evidence, "evidence_id", "foreign-evidence")
        case "inventory":
            match evidence:
                case RegionMeshingEvidence():
                    object.__setattr__(
                        evidence, "cell_global_ids", evidence.cell_global_ids[1:]
                    )
                case RegionBoundaryEvidence():
                    scope = evidence.boundary_label.scope
                    changed = eqx.tree_at(
                        lambda value: value.global_entity_ids,
                        scope,
                        scope.global_entity_ids + 100,
                    )
                    object.__setattr__(
                        evidence,
                        "boundary_label",
                        phx.meshing.MeshLabel(evidence.boundary_label.name, changed),
                    )
    with pytest.raises(
        ValueError,
        match="must retain matching evidence, mesh, topology, and local inventory",
    ):
        reconstruct_meshing_source(evidence, results=(region_owner,))


def test_region_evidence_binding_revalidates_complete_stored_content(
    region_owner: CellMeshingResult,
) -> None:
    evidence = copy(_evidence(region_owner))
    object.__setattr__(evidence, "evidence_id", "forged-retained-evidence")
    owner = copy(region_owner)
    match evidence:
        case RegionMeshingEvidence():
            object.__setattr__(owner, "region_evidence", evidence)
        case RegionBoundaryEvidence():
            object.__setattr__(owner, "region_boundary_evidence", (evidence,))
    with pytest.raises(ValueError, match="must reproduce its stored content"):
        reconstruct_meshing_source(evidence, results=(owner,))


def test_region_evidence_without_retained_result_refuses(
    region_owner: CellMeshingResult,
) -> None:
    with pytest.raises(ValueError, match="no matching CellMeshingResult"):
        reconstruct_meshing_source(_evidence(region_owner), meshes=(region_owner.mesh,))

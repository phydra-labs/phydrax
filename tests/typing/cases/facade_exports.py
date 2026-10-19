"""Formerly scanned facade exports retain their defining types."""

from typing import assert_type

from phydrax import SpatialCoordinateContract
from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    prepare_surface_chart_deformation,
    PreparedSurfaceChartDeformation,
    SurfaceChartWitness,
    SurfaceGeometryReconstruction,
)
from phydrax.geometry import (
    SphereMaterialCellAtlas,
)
from phydrax.meshing import MeshingDomain
from phydrax.typing import ConvertibleToArray


def surface_chart_exports(
    source_mesh: CellMesh,
    geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    reconstruction: SurfaceGeometryReconstruction,
    domain: MeshingDomain,
    source_witness: SurfaceChartWitness,
    target_witness: SurfaceChartWitness,
) -> None:
    assert_type(source_witness, SurfaceChartWitness)
    assert_type(target_witness, SurfaceChartWitness)
    prepared = prepare_surface_chart_deformation(
        source_mesh,
        geometry,
        target_mesh,
        reconstruction,
        domain,
        source_witness=source_witness,
        target_witness=target_witness,
        maximum_fidelity=1e-8,
        maximum_displacement=1e-8,
    )
    assert_type(prepared, PreparedSurfaceChartDeformation)


def sphere_atlas_exports(
    domain: MeshingDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    contract: SpatialCoordinateContract,
    patches: ConvertibleToArray,
    dimensions: ConvertibleToArray,
    indices: ConvertibleToArray,
    parameters: ConvertibleToArray,
    occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
    entity_ids: tuple[tuple[str, ...], ...],
) -> None:
    atlas = SphereMaterialCellAtlas(
        domain,
        mesh,
        geometry,
        contract,
        cell_patches=patches,
        corner_source_dimensions=dimensions,
        corner_source_indices=indices,
        corner_source_parameters=parameters,
        corner_source_occurrence_paths=occurrence_paths,
        corner_source_entity_ids=entity_ids,
        maximum_fidelity=1e-8,
    )
    assert_type(atlas, SphereMaterialCellAtlas)

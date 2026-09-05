import numpy as np
import pytest

import phydrax as phx


def _scope(dimension=2):
    return phx.meshing.MeshingScope(
        "shape",
        "revision-1",
        phx.meshing.MeshingEntityKind.GEOMETRY,
        dimension,
        f"entities-{dimension}",
        np.asarray([10, 20], dtype=np.int64),
    )


def test_surface_and_volume_specs_are_exact_and_identity_bearing():
    surface_scope = _scope()
    surface_target = phx.meshing.CellMeshingTarget(
        2,
        3,
        phx.meshing.CellFamilyPolicy(required=("triangle",)),
    )
    size = phx.meshing.UniformSizeControl(surface_scope, 0.1)
    surface = phx.meshing.SurfaceMeshingSpec(
        surface_target,
        surface_scope,
        size_controls=(size,),
    )

    volume_scope = _scope(3)
    volume_target = phx.meshing.CellMeshingTarget(
        3,
        3,
        phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
    )
    volume_size = phx.meshing.UniformSizeControl(volume_scope, 0.2)
    volume = phx.meshing.VolumeMeshingSpec(
        volume_target,
        volume_scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(volume_size,),
    )

    assert surface.specification_id != volume.specification_id
    assert surface.target.topological_dimension == 2
    assert volume.target.topological_dimension == 3


def test_provider_support_report_fails_before_execution():
    scope = _scope()
    target = phx.meshing.CellMeshingTarget(
        2,
        3,
        phx.meshing.CellFamilyPolicy(required=("triangle",)),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        target,
        scope,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.1),),
    )
    provider = phx.meshing.MeshingProviderInfo(
        "unit-provider",
        "1",
        "MIT",
        operations=(phx.meshing.MeshingOperation.MESH_SURFACE,),
        source_kinds=(phx.meshing.MeshingSourceKind.BREP,),
        capabilities=(),
        cell_kinds=("triangle",),
        dimensions=(2,),
        execution_modes=(phx.meshing.MeshingExecutionMode.IN_PROCESS,),
    )
    source = phx.meshing.MeshingSourceDescriptor(
        "shape",
        "revision-1",
        phx.meshing.MeshingSourceKind.BREP,
        2,
        3,
        closed=False,
    )
    report = phx.meshing.ProviderSupportReport(
        provider,
        source,
        specification,
        unsupported=("periodic surface meshing",),
    )

    with pytest.raises(phx.meshing.MeshingFailure) as captured:
        report.require_supported()
    assert (
        captured.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )

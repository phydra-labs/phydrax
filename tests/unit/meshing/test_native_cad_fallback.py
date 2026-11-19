from __future__ import annotations

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep import brep_box
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from tools.meshing_qualification.scenarios.native_family_matrix import (
    _CONTRACT,
    _quarter_surface_case,
)


pytestmark = pytest.mark.meshcore


def _assert_certified_fallback(plan: phx.meshing.NativeMeshingPlan) -> None:
    candidate = plan.execute(prove=False)
    assert candidate.certified_steps == ("entire_certified_route",)
    assert not candidate.screens
    proved = candidate.prove()
    direct = plan.execute(prove=True)
    assert proved.mesh.mesh_id == direct.mesh.mesh_id
    assert proved.certification is not None
    assert direct.certification is not None
    assert proved.certification.report_id == direct.certification.report_id


def test_quad_surface_uses_certified_fallback() -> None:
    case = _quarter_surface_case(1)
    plan = NativeMeshingProvider(case.options).plan(
        case.source, case.specification, coordinate_contract=_CONTRACT
    )
    _assert_certified_fallback(plan)


def test_layered_box_uses_certified_fallback() -> None:
    contract = phx.SpatialCoordinateContract.si()
    model = brep_box(np.zeros(3), np.ones(3), coordinate_contract=contract)
    boundary = phx.meshing.MeshingScope.whole(model, 2)
    control = phx.meshing.BoundaryLayerControl(
        boundary,
        phx.meshing.LayerSchedule((0.01, 0.015)),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        volume_scope=phx.meshing.MeshingScope.whole(model, 3),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allowed_transitions=("pyramid",),
                allow_mixed=True,
            ),
        ),
        boundary,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                boundary, 0.34, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        layer_controls=(control,),
        maximum_deviation=0.05,
    )
    options = NativeMeshingOptions(
        "cad_tetrahedral",
        volume_schedule=phx.meshing.NativeVolumeSchedule(
            radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
        ),
    )
    plan = NativeMeshingProvider(options).plan(model, specification)
    _assert_certified_fallback(plan)


def test_multiregion_box_uses_certified_fallback() -> None:
    pytest.importorskip("OCP")
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    splitter = BOPAlgo_Splitter()
    for x in (0.0, 1.0):
        splitter.AddArgument(
            BRepPrimAPI_MakeBox(gp_Pnt(x, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
        )
    splitter.Perform()
    assert not splitter.HasErrors()
    model = phx.interchange.import_occt_shape(
        splitter.Shape(), coordinate_contract=phx.SpatialCoordinateContract.si()
    ).model
    boundary = phx.meshing.MeshingScope.whole(model, 2)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        boundary,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                boundary, 0.5, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        maximum_deviation=0.05,
    )
    options = NativeMeshingOptions(
        "cad_tetrahedral",
        volume_schedule=phx.meshing.NativeVolumeSchedule(
            radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
        ),
    )
    plan = NativeMeshingProvider(options).plan(model, specification)
    _assert_certified_fallback(plan)

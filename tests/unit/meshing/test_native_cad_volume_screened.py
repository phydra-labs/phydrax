from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_identity
from phydrax.geometry.brep._constructors import brep_box
from phydrax.meshing._canonical import _accept_native_construction, _NATIVE_PROVIDER
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native import _PreparedDirectBRepCadVolume
from phydrax.meshing.providers._native_cad_volume import _screen_cad_shell


pytestmark = pytest.mark.meshcore


def _box_plan() -> phx.meshing.NativeMeshingPlan:
    contract = phx.SpatialCoordinateContract.si()
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=contract,
        source_id="screened-cad-box",
    )
    scope = phx.meshing.MeshingScope.whole(model, 2)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 0.5, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        protected_features=(
            phx.meshing.ProtectedFeature(
                scope, phx.meshing.FeatureKind.SURFACE, maximum_deviation=0.02
            ),
        ),
        maximum_deviation=0.02,
    )
    options = NativeMeshingOptions(
        "cad_tetrahedral",
        volume_schedule=phx.meshing.NativeVolumeSchedule(
            shape_strength=phx.meshing.SizeControlStrength.SOFT
        ),
    )
    return NativeMeshingProvider(options).plan(
        model, specification, coordinate_contract=contract
    )


def test_screened_cad_volume_can_be_proved_without_remeshing() -> None:
    plan = _box_plan()
    assert isinstance(plan.prepared, _PreparedDirectBRepCadVolume)
    screened = plan.prepared.execute(steering="screened")
    certified = plan.prepared.execute(steering="certified")
    assert screened.deferred_certification is not None
    assert screened.compliance.issues == (
        "compliance not assessed: screened construction",
    )
    assert screened.certification.fidelity_witness is None
    assert screened.certification.scoped_fidelity == ()
    assert np.array_equal(
        np.asarray(screened.mesh.coordinates), np.asarray(certified.mesh.coordinates)
    )
    assert all(
        np.array_equal(np.asarray(left.vertices), np.asarray(right.vertices))
        for left, right in zip(screened.mesh.blocks, certified.mesh.blocks, strict=True)
    )
    tetrahedra = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in screened.mesh.blocks]
    )
    faces = Counter(
        tuple(sorted(int(v) for v in tet[list(face)]))
        for tet in tetrahedra
        for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    )
    assert all(count in (1, 2) for count in faces.values())
    exterior = [face for face, count in faces.items() if count == 1]
    edges = Counter(
        tuple(sorted((face[i], face[(i + 1) % 3]))) for face in exterior for i in range(3)
    )
    assert all(count == 2 for count in edges.values())
    resolved = screened.deferred_certification.resolve()
    assert resolved.mesh is screened.mesh
    assert resolved.geometry is screened.geometry
    assert resolved.deferred_certification is None
    proved = _accept_native_construction(
        resolved,
        plan.specification,
        _NATIVE_PROVIDER,
        meshcore_identity(),
        plan.plan_id,
        plan.route,
    )
    expected = _accept_native_construction(
        certified,
        plan.specification,
        _NATIVE_PROVIDER,
        meshcore_identity(),
        plan.plan_id,
        plan.route,
    )
    assert proved.mesh.mesh_id == expected.mesh.mesh_id
    assert tuple(a.association_id for a in proved.associations) == tuple(
        a.association_id for a in expected.associations
    )


def test_screen_refuses_open_or_misoriented_shell() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    facets = np.asarray(((-1, 0),), dtype=np.int64)
    closed = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    _screen_cad_shell(points, closed, [0] * len(closed), facets)
    with pytest.raises(ValueError, match="closed consistently oriented region edges"):
        _screen_cad_shell(points, closed[:-1], [0] * (len(closed) - 1), facets)
    reversed_face = [closed[0][::-1], *closed[1:]]
    with pytest.raises(ValueError, match="closed consistently oriented region edges"):
        _screen_cad_shell(points, reversed_face, [0] * len(reversed_face), facets)


def test_screen_checks_each_region_at_a_three_face_junction() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    triangles = [
        (0, 2, 1),
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
        (0, 2, 4),
        (2, 1, 4),
        (1, 0, 4),
    ]
    facets = np.asarray(((1, 0), (-1, 0), (-1, 1)), dtype=np.int64)
    owners = [0, 1, 1, 1, 2, 2, 2]
    _screen_cad_shell(points, triangles, owners, facets)
    with pytest.raises(ValueError, match="closed consistently oriented region edges"):
        _screen_cad_shell(points, [triangles[0][::-1], *triangles[1:]], owners, facets)

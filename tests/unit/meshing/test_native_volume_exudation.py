"""The explicit weighted sliver-exudation stage of native PLC volume generation.

Oracles are independent of the mesher: exact orientation, oriented face
pairing, analytic boundary area and volume, the immutable PLC corners, and the
publishing cell-validity audit.
"""

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    exact_orient3d,
    meshcore_available,
    MeshcoreStatus,
    TET_MESH_EXUDE_COUNTERS,
    TET_MESH_IMPROVE_COUNTERS,
    TET_MESH_REFINE_COUNTERS,
    TET_MESH_UNMET_CRITERIA,
)
from phydrax.meshing._volume_generation import (
    generate_plc_volume,
    NativeVolumeSchedule,
    VolumeConstruction,
)
from phydrax.meshing.providers._native_volume import plc_volume_audit_policy


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)

M = phx.meshing
_SI = phx.SpatialCoordinateContract.si()
_BOX_CORNERS = np.asarray(
    [[(i >> axis) & 1 for axis in range(3)] for i in range(8)],
    dtype=np.float64,
)
_CORNERS = np.concatenate(
    (
        _BOX_CORNERS,
        np.asarray(([0.5, 0.5, 0.001], [0.5, 0.5, 0.999]), dtype=np.float64),
    )
)
_LOOPS = (
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
)
_COMPLEX = M.PiecewiseLinearComplex(_CORNERS, _LOOPS, np.zeros(6), [[-1, 0]], ("solid",))
_SOURCE = M.NativePlcSource(_COMPLEX, "exudation-box", "r1")


def _schedule(exudation_passes: int) -> NativeVolumeSchedule:
    # A demanding dihedral aim after one improvement sweep leaves interior slivers.
    return NativeVolumeSchedule(
        minimum_dihedral_degrees=25.0,
        improvement_passes=1,
        exudation_passes=exudation_passes,
        exudation_weight_fraction=0.25,
    )


_SCHEDULE = _schedule(4)


def _specification(limits: M.MeshingLimits | None = None) -> M.VolumeMeshingSpec:
    scope = M.MeshingScope(
        _SOURCE.source_id,
        _SOURCE.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "exudation-box-facets",
        np.arange(_COMPLEX.facet_count, dtype=np.int64),
    )
    return M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, 0.25, strength=M.SizeControlStrength.SOFT),
        ),
        limits=limits,
    )


def _generate(
    schedule: NativeVolumeSchedule, limits: M.MeshingLimits | None = None
) -> VolumeConstruction:
    return generate_plc_volume(
        _COMPLEX,
        _specification(limits),
        schedule,
        validity_policy=plc_volume_audit_policy().validity_policy,
        source_id=_SOURCE.source_id,
        source_revision=_SOURCE.source_revision,
        input_id="exudation",
    )


def _removal_schedule(
    extra_points: tuple[tuple[float, float, float], ...],
    size: float,
    minimum_dihedral: float,
    passes: int,
    source_id: str,
    protect_points: bool,
    /,
) -> VolumeConstruction:
    points = np.concatenate((_BOX_CORNERS, np.asarray(extra_points, dtype=np.float64)))
    complex_ = M.PiecewiseLinearComplex(
        points, _LOOPS, np.zeros(6), [[-1, 0]], ("solid",)
    )
    scope = M.MeshingScope(
        source_id,
        "authored",
        M.MeshingEntityKind.GEOMETRY,
        2,
        f"{source_id}-facets",
        np.arange(6, dtype=np.int64),
    )
    protected = M.MeshingScope(
        source_id,
        "authored",
        M.MeshingEntityKind.GEOMETRY,
        0,
        f"{source_id}-protected-points",
        np.arange(8, points.shape[0], dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, size, strength=M.SizeControlStrength.SOFT),
        ),
        protected_features=(
            (M.ProtectedFeature(protected, M.FeatureKind.CORNER),)
            if protect_points
            else ()
        ),
    )
    schedule = NativeVolumeSchedule(
        minimum_dihedral_degrees=minimum_dihedral,
        improvement_passes=passes,
    )
    return generate_plc_volume(
        complex_,
        specification,
        schedule,
        validity_policy=plc_volume_audit_policy().validity_policy,
        source_id=source_id,
        source_revision="authored",
        input_id=source_id,
    )


def _assert_independent_unit_box(
    construction: VolumeConstruction,
    source_points: np.ndarray = _BOX_CORNERS,
) -> None:
    points = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    cells = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
    corners = points[cells]
    assert np.all(
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]) > 0
    )
    opposite = ((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2))
    oriented = np.concatenate([cells[:, list(face)] for face in opposite])
    _, inverse, counts = np.unique(
        np.sort(oriented, axis=1),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    assert np.all(counts <= 2)
    boundary = points[oriented[counts[inverse] == 1]]
    area = 0.5 * np.linalg.norm(
        np.cross(boundary[:, 1] - boundary[:, 0], boundary[:, 2] - boundary[:, 0]),
        axis=1,
    )
    assert np.sum(area) == pytest.approx(6.0, rel=1e-12)
    volume = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    assert np.sum(volume) == pytest.approx(1.0, rel=1e-12)
    for corner in source_points:
        assert np.any(np.all(points == corner, axis=1))
    validity = phx.discretization.certify_cell_geometry_validity(
        phx.discretization.CellGeometrySpec.affine(construction.mesh),
        mesh=construction.mesh,
        policy=plc_volume_audit_policy().validity_policy,
    )
    assert validity.invalid_count == 0 and validity.unresolved_count == 0


def test_explicit_exudation_stage_weights_only_interior_vertices_and_keeps_the_domain() -> (
    None
):
    original_points = np.asarray(_COMPLEX.vertices).copy()
    original_loops = np.asarray(_COMPLEX.polygon_vertices).copy()
    disabled = _generate(_schedule(0))
    assert disabled.exudation is None and disabled.exudation_weights is None
    construction = _generate(_SCHEDULE)
    run = construction.exudation
    assert run is not None
    assert run.status in (MeshcoreStatus.OK, MeshcoreStatus.REFINEMENT_LIMIT)
    counters = dict(zip(TET_MESH_EXUDE_COUNTERS, run.counters.tolist(), strict=True))
    assert (
        counters["flips"] > 0
        and counters["weighted_vertices"] > 0
        and counters["passes"] >= 1
    )
    # Exudation consumed the same refined/improved carrier as the disabled run.
    assert np.array_equal(construction.refinement.counters, disabled.refinement.counters)
    assert np.array_equal(
        construction.improvement.counters, disabled.improvement.counters
    )
    points = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    weights = construction.exudation_weights
    assert weights is not None and weights.shape == (points.shape[0],)
    assert np.all(weights >= 0.0)
    on_boundary = np.any((points == 0.0) | (points == 1.0), axis=1)
    assert np.all(weights[on_boundary] == 0.0)
    assert np.count_nonzero(weights) == counters["weighted_vertices"]
    # Every accepted move strictly raises its cavity's smallest dihedral angle.
    assert construction.quality.minimum_dihedral >= disabled.quality.minimum_dihedral
    assert all(
        criterion in TET_MESH_UNMET_CRITERIA for criterion, _, _ in construction.unmet
    )
    _assert_independent_unit_box(construction)
    # Unprotected interior seeds may retire; their authored source bank may not.
    np.testing.assert_array_equal(_COMPLEX.vertices, original_points)
    np.testing.assert_array_equal(_COMPLEX.polygon_vertices, original_loops)
    fidelity = construction.source_fidelity
    initial_fidelity = disabled.source_fidelity
    assert fidelity is not None and initial_fidelity is not None
    source = fidelity.declared_source
    initial_source = initial_fidelity.declared_source
    assert source is not None and initial_source is not None
    np.testing.assert_array_equal(source.points, original_points)
    for name in (
        "points",
        "faces",
        "face_ids",
        "face_group_rows",
        "face_tolerances",
        "segments",
        "segment_ids",
        "segment_group_rows",
        "segment_tolerances",
    ):
        np.testing.assert_array_equal(
            getattr(source, name),
            getattr(initial_source, name),
        )
    assert fidelity.achieved_bound == 0.0
    assert np.all(fidelity.witness_deviations == 0.0)
    assert np.array_equal(fidelity.facet_achieved, initial_fidelity.facet_achieved)


@pytest.mark.parametrize(
    (
        "extra_points",
        "size",
        "minimum_dihedral",
        "passes",
        "counter",
        "protect_points",
        "expected",
    ),
    (
        (
            (
                (0.0760947627385314, 0.41197613239012343, 0.6155840603007043),
                (0.906988555512639, 0.1120128918093043, 0.9235640156222432),
            ),
            0.4,
            35.0,
            5,
            "multiface_removals",
            True,
            1,
        ),
        (
            (
                (0.8246849216269895, 0.8721394949038036, 0.12160955151634446),
                (0.10402896037966443, 0.8764292325293658, 0.7159833463772252),
                (0.7460274786731873, 0.5675077474325294, 0.4234190735295684),
                (0.16447686126850444, 0.6455119418296266, 0.7558712100722097),
            ),
            0.45,
            40.0,
            8,
            "vertex_removals",
            False,
            1,
        ),
        (
            (
                (0.8246849216269895, 0.8721394949038036, 0.12160955151634446),
                (0.10402896037966443, 0.8764292325293658, 0.7159833463772252),
                (0.7460274786731873, 0.5675077474325294, 0.4234190735295684),
                (0.16447686126850444, 0.6455119418296266, 0.7558712100722097),
            ),
            0.45,
            40.0,
            8,
            "vertex_removals",
            True,
            0,
        ),
    ),
    ids=("multiface", "vertex", "protected-vertex"),
)
def test_volume_schedule_consumes_multiface_and_vertex_removal(
    extra_points: tuple[tuple[float, float, float], ...],
    size: float,
    minimum_dihedral: float,
    passes: int,
    counter: str,
    protect_points: bool,
    expected: int,
) -> None:
    construction = _removal_schedule(
        extra_points,
        size,
        minimum_dihedral,
        passes,
        f"{counter}-schedule",
        protect_points,
    )
    counters = dict(
        zip(
            TET_MESH_IMPROVE_COUNTERS,
            construction.improvement.counters.tolist(),
            strict=True,
        )
    )
    if expected:
        assert counters[counter] >= expected
        assert counters["attempts"] > counters[counter]
    else:
        assert counters[counter] == 0
    points = np.asarray(construction.mesh.coordinates)
    if protect_points:
        for protected_point in extra_points:
            assert np.any(np.all(points == protected_point, axis=1))
    source_points = (
        np.concatenate((_BOX_CORNERS, np.asarray(extra_points)))
        if protect_points
        else _BOX_CORNERS
    )
    _assert_independent_unit_box(construction, source_points)


def test_provider_publishes_exudation_evidence_with_certified_coverage() -> None:
    options = M.NativeMeshingOptions("plc_tetrahedral", volume_schedule=_SCHEDULE)
    result = (
        M.NativeMeshingProvider(options)
        .plan(
            _SOURCE,
            _specification(),
            coordinate_contract=_SI,
        )
        .execute()
    )
    assert result.audit.passed
    assert result.certification is not None and result.certification.passed
    achieved = dict(result.compliance.achieved)
    direct = _generate(_SCHEDULE).exudation
    assert direct is not None
    for name, value in zip(
        TET_MESH_EXUDE_COUNTERS, direct.counters.tolist(), strict=True
    ):
        assert achieved[f"exudation:{name}"] == value


def test_exudation_without_remaining_work_is_a_resource_refusal() -> None:
    disabled = _generate(_schedule(0))
    spent = (
        sum(
            int(value)
            for name, value in disabled.construction_counters
            if name.endswith("work_units")
        )
        + int(disabled.refinement.counters[TET_MESH_REFINE_COUNTERS.index("work_units")])
        + int(
            disabled.improvement.counters[TET_MESH_IMPROVE_COUNTERS.index("work_units")]
        )
    )
    # One weighted trial (16 units) fits; the stage cannot finish.
    with pytest.raises(M.MeshingFailure) as refusal:
        _generate(_SCHEDULE, M.MeshingLimits(maximum_work_units=spent + 17))
    assert refusal.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert "exudation" in str(refusal.value)
    assert set(TET_MESH_EXUDE_COUNTERS) <= set(dict(refusal.value.evidence.achieved))


def test_exudation_controls_are_validated_and_identify_the_schedule() -> None:
    assert NativeVolumeSchedule().exudation_passes == 0
    assert (
        NativeVolumeSchedule(exudation_passes=1).schedule_id
        != NativeVolumeSchedule().schedule_id
    )
    with pytest.raises(ValueError, match="exudation_weight_fraction"):
        NativeVolumeSchedule(exudation_weight_fraction=0.3)
    with pytest.raises(TypeError, match="exudation_passes"):
        NativeVolumeSchedule(exudation_passes=True)

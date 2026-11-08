from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from phydrax._meshcore import (
    MeshcoreStatus,
    PLC_SEARCH_BUDGET,
    PlcRecovery3D,
    PlcSearchBudget,
    prepare_plc_source,
)
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._domain_certification import _upper_sum, certify_domain_coverage
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._planar_coverage import PlcSourcePointWitness
from phydrax.meshing._association import (
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._certification_inputs import _prepare_region_assignments
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import (
    FeatureKind,
    ProtectedFeature,
    RegionControl,
    RegionSeed,
)
from phydrax.meshing._native_construction import NativeMeshConstruction
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex, PlcBoundaryPolicy
from phydrax.meshing._result import _validated_source_evidence, CellMeshingResult
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import (
    SizeCombinationPolicy,
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing._volume_generation import (
    _root_normalized_candidate,
    _source_face_ids,
    _source_tables,
    compose_plc_root_point,
    generate_plc_volume,
    NativeVolumeSchedule,
    VolumeConstruction,
)
from phydrax.meshing.providers._native import NativeMeshingProvider
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import NativePlcSource
from phydrax.meshing.providers._native_volume import (
    _organization,
    _source_associations,
    _volume_compliance,
    prepare_native_plc_volume,
)


pytestmark = pytest.mark.meshcore


@pytest.mark.parametrize(
    ("first", "second"),
    (
        (1e-20, 1.0),
        (1e-17, 1.0),
        (8.673617379884035e-19, 1.0),
        (1.0, 1e-20),
        (0.1, 3.0),
    ),
    ids=("win5-c4-1e-20", "win5-c4-1e-17", "win5-c4-2^-60", "ordered", "rounded"),
)
def test_source_distance_upper_sum_encloses_the_exact_sum(
    first: float, second: float
) -> None:
    # WIN-5 C4: a Fast2Sum test without ordering returned 1.0 for 1e-20 + 1.0.
    exact = Fraction(first) + Fraction(second)
    total = _upper_sum(first, second)
    assert Fraction(total) >= exact
    assert Fraction(float(np.nextafter(total, -np.inf))) < exact


def test_composed_plc_root_point_rounds_once_and_rejects_bad_stencils() -> None:
    half = Fraction(1, 2)
    one = Fraction(1)
    next_one = Fraction(float(np.nextafter(1.0, 2.0)))
    roots = ((one, Fraction(0), Fraction(0)), (next_one, Fraction(0), Fraction(0)))
    stencils = (((0, half), (1, half)), ((1, one),))
    terms, point = compose_plc_root_point(roots, stencils, (0, 1), (half, half))
    assert terms == ((0, Fraction(1, 4)), (1, Fraction(3, 4)))
    assert float(point[0]) == float(next_one)
    rounded_midpoint = float((one + next_one) / 2)
    assert float((Fraction(rounded_midpoint) + next_one) / 2) != float(point[0])
    with pytest.raises(ValueError, match="coefficients"):
        compose_plc_root_point(roots, stencils, (0, 1), (one, half))
    with pytest.raises(ValueError, match="stencils"):
        compose_plc_root_point(roots, (((0, one), (0, one)),), (0,), (one,))


def test_root_normalization_preserves_input_signed_zero_and_rejects_change() -> None:
    points = np.asarray(((-0.0, 1.0, 2.0),), dtype=np.float64)
    arrays = SimpleNamespace(
        points=points,
        faces=np.empty((0, 3), dtype=np.int32),
        segments=np.empty((0, 2), dtype=np.int32),
    )
    roots = ((Fraction(0), Fraction(1), Fraction(2)),)
    stencils = (((0, Fraction(1)),),)
    normalized, terms, exact = _root_normalized_candidate(
        cast(Any, arrays), cast(Any, None), roots, stencils
    )
    np.testing.assert_array_equal(normalized.view(np.uint64), points.view(np.uint64))
    assert terms == stencils
    assert exact == roots
    arrays.points = np.asarray(((0.25, 1.0, 2.0),), dtype=np.float64)
    with pytest.raises(ValueError, match="input vertex"):
        _root_normalized_candidate(cast(Any, arrays), cast(Any, None), roots, stencils)


def test_mapped_atomic_size_banks_require_exact_cavity_and_source_targets() -> None:
    from phydrax.meshing._layer_core_mapped import _prove_mapped_atomic_size_banks

    old_points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    new_points = np.vstack((old_points, (0.25, 0.25, 0.25)))
    old_ids = np.asarray((0, 1, 2, 3), dtype=np.int64)
    new_ids = np.asarray((0, 1, 2, 3, 4), dtype=np.int64)
    old_tets = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    new_tets = np.asarray(
        ((4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)), dtype=np.int64
    )
    args = [
        old_ids,
        old_points,
        old_tets,
        np.asarray((10,)),
        np.asarray((0,)),
        np.full(4, 2.0),
        np.asarray((2.0,)),
        new_ids,
        new_points,
        new_tets,
        np.asarray((11, 12, 13, 14)),
        np.zeros(4, dtype=np.int64),
        np.zeros(6, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    ]
    sizes, cavities = _prove_mapped_atomic_size_banks(*args)
    np.testing.assert_array_equal(sizes, np.full(5, 2.0))
    assert cavities == (((10,), (11, 12, 13, 14)),)

    wrong_field = list(args)
    wrong_field[5] = np.full(4, 0.1)
    with pytest.raises(ValueError, match="contradicts"):
        _prove_mapped_atomic_size_banks(*wrong_field)
    missing_cavity = list(args)
    missing_cavity[9], missing_cavity[10], missing_cavity[11] = (
        new_tets[:-1],
        np.asarray((11, 12, 13)),
        np.zeros(3, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="oriented material boundary"):
        _prove_mapped_atomic_size_banks(*missing_cavity)
    outside = list(args)
    outside[8] = new_points.copy()
    outside[8][4] = (2.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="lacks an exact original reference cell"):
        _prove_mapped_atomic_size_banks(*outside)

    shared_points = np.vstack((old_points, (0.0, 0.0, -1.0)))
    ambiguous = list(args)
    ambiguous[0] = np.asarray((0, 1, 2, 3, 5))
    ambiguous[1] = shared_points
    ambiguous[2] = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)))
    ambiguous[3] = np.asarray((10, 15))
    ambiguous[4] = np.asarray((0, 0))
    ambiguous[5] = np.asarray((2.0, 2.0, 2.0, 2.0, 0.0))
    ambiguous[6] = np.asarray((2.0, 0.0))
    ambiguous[8] = new_points.copy()
    ambiguous[8][4] = (0.25, 0.25, 0.0)
    with pytest.raises(ValueError, match="ambiguous original-cell targets"):
        _prove_mapped_atomic_size_banks(*ambiguous)


_POINTS = np.asarray(
    (
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ),
    dtype=np.float64,
)

_POLYGONS = np.asarray(
    (
        (1, 2, 3),
        (0, 3, 2),
        (0, 1, 3),
        (2, 1, 4),
        (0, 4, 1),
        (0, 2, 4),
        (0, 1, 2),
    ),
    dtype=np.int64,
)

_FACET_REGIONS = np.asarray(
    (
        (-1, 0),
        (-1, 0),
        (-1, 0),
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (0, 1),
    ),
    dtype=np.int64,
)


def _source(boundary: PlcBoundaryPolicy = "fixed") -> NativePlcSource:
    complex_ = PiecewiseLinearComplex(
        _POINTS,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"vertex-{row}" for row in range(_POINTS.shape[0])),
        vertex_occurrence_paths=((),) * _POINTS.shape[0],
        facet_ids=tuple(f"facet-{row}" for row in range(_POLYGONS.shape[0])),
        facet_occurrence_paths=((),) * _POLYGONS.shape[0],
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
        boundary=boundary,
    )
    return NativePlcSource(complex_, "bipyramid", "revision-1")


def _grouped_cube_source(
    boundary: PlcBoundaryPolicy = "fixed", angle: float | None = None
) -> NativePlcSource:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    if angle is not None:
        # A cube of side 1/3 centred at the origin, rotated so that the two
        # authored triangles of a face are not exactly coplanar for angle != 0.
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.asarray(((c, -s, 0.0), (s * c, c * c, -s), (s * s, s * c, c)))
        points = (points - 0.5) @ rotation.T / 3.0
    triangles = np.asarray(
        (
            (0, 2, 3),
            (0, 3, 1),
            (4, 5, 7),
            (4, 7, 6),
            (0, 1, 5),
            (0, 5, 4),
            (2, 6, 7),
            (2, 7, 3),
            (0, 4, 6),
            (0, 6, 2),
            (1, 3, 7),
            (1, 7, 5),
        ),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        points,
        triangles,
        np.repeat(np.arange(6, dtype=np.int64), 2),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (6, 1)),
        ("solid",),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"cube-vertex-{row}" for row in range(points.shape[0])),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=tuple(f"cube-facet-{row}" for row in range(6)),
        facet_occurrence_paths=((),) * 6,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary=boundary,
    )
    return NativePlcSource(complex_, "grouped-cube", "revision-1")


def _repeated_region_source() -> NativePlcSource:
    base = _source()
    complex_ = PiecewiseLinearComplex(
        _POINTS,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("shared-region", "shared-region"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=base.complex.vertex_ids,
        vertex_occurrence_paths=base.complex.vertex_occurrence_paths,
        facet_ids=base.complex.facet_ids,
        facet_occurrence_paths=base.complex.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=(("assembly-a",), ("assembly-b",)),
        boundary=base.complex.boundary,
    )
    return NativePlcSource(complex_, "repeated-bipyramid", "revision-1")


def _request(
    source: NativePlcSource,
    /,
    *,
    region_controls: tuple[RegionControl, ...] = (),
    region_seeds: tuple[RegionSeed, ...] | None = None,
) -> VolumeMeshingSpec:
    boundary = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    seeds = (
        (
            RegionSeed(
                np.asarray((0.0, 0.25, 0.25), dtype=np.float64),
                "upper",
                "upper-material",
                RegionRole.SOLID,
            ),
            RegionSeed(
                np.asarray((0.0, 0.25, -0.25), dtype=np.float64),
                "lower",
                "lower-material",
                RegionRole.SOLID,
            ),
        )
        if region_seeds is None
        else region_seeds
    )
    return VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        region_controls=region_controls,
        region_seeds=seeds,
    )


def _schedule() -> NativeVolumeSchedule:
    return NativeVolumeSchedule(
        radius_edge_bound=8.0,
        minimum_dihedral_degrees=0.0,
    )


def test_native_volume_schedule_omits_only_requested_radius_target() -> None:
    from phydrax.meshing._model_codec import restore_model_archive_value

    default = NativeVolumeSchedule()
    explicit = NativeVolumeSchedule(radius_edge_bound=2.0, minimum_dihedral_degrees=10.0)
    no_radius = NativeVolumeSchedule(radius_edge_bound=None, minimum_dihedral_degrees=0.0)
    assert default.schedule_id == explicit.schedule_id
    assert no_radius.radius_edge_bound is None
    assert no_radius.minimum_dihedral_degrees == 0.0
    assert no_radius.minimum_relative_determinant == default.minimum_relative_determinant
    assert no_radius.schedule_id != default.schedule_id
    rebuilt = restore_model_archive_value(no_radius)
    assert isinstance(rebuilt, NativeVolumeSchedule)
    assert rebuilt.schedule_id == no_radius.schedule_id
    assert rebuilt.radius_edge_bound is None
    for invalid in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="radius_edge_bound"):
            NativeVolumeSchedule(radius_edge_bound=invalid)


def test_native_volume_shape_strength_keeps_hard_identity() -> None:
    from phydrax._fingerprint import canonical_fingerprint
    from phydrax.meshing._model_codec import restore_model_archive_value

    default = NativeVolumeSchedule()
    hard = NativeVolumeSchedule(shape_strength=SizeControlStrength.HARD)
    soft = NativeVolumeSchedule(shape_strength=SizeControlStrength.SOFT)
    legacy_payload: dict[str, object] = {
        "kind": "native-volume-schedule",
        "radius_edge_bound": 2.0,
        "minimum_dihedral_degrees": 10.0,
        "minimum_relative_determinant": default.minimum_relative_determinant,
    }
    assert default.shape_strength is SizeControlStrength.HARD
    # The pre-strength fingerprint: hard schedules keep their exact identity.
    assert (
        default.schedule_id == hard.schedule_id == canonical_fingerprint(legacy_payload)
    )
    assert soft.schedule_id == canonical_fingerprint(
        {**legacy_payload, "shape_strength": "soft"}
    )

    def options(schedule: NativeVolumeSchedule | None) -> NativeMeshingOptions:
        return NativeMeshingOptions("plc_tetrahedral", volume_schedule=schedule)

    assert options(hard).options_id == options(None).options_id
    assert options(soft).options_id != options(hard).options_id
    rebuilt = restore_model_archive_value(soft)
    assert isinstance(rebuilt, NativeVolumeSchedule)
    assert rebuilt.shape_strength is SizeControlStrength.SOFT
    assert rebuilt.schedule_id == soft.schedule_id
    with pytest.raises(TypeError, match="shape_strength"):
        NativeVolumeSchedule(shape_strength=cast(Any, "soft"))


def test_native_volume_schedule_declares_the_plc_search_budget() -> None:
    from phydrax._fingerprint import canonical_fingerprint
    from phydrax.meshing._model_codec import restore_model_archive_value

    default = NativeVolumeSchedule()
    explicit = NativeVolumeSchedule(plc_search_budget=PLC_SEARCH_BUDGET)
    declared = NativeVolumeSchedule(plc_search_budget=PlcSearchBudget(1 << 20, 1 << 28))
    payload: dict[str, object] = {
        "kind": "native-volume-schedule",
        "radius_edge_bound": 2.0,
        "minimum_dihedral_degrees": 10.0,
        "minimum_relative_determinant": default.minimum_relative_determinant,
    }
    # The documented default keeps the schedule identity; a declared budget
    # is part of it.
    assert default.plc_search_budget == PLC_SEARCH_BUDGET
    assert default.schedule_id == explicit.schedule_id == canonical_fingerprint(payload)
    assert declared.schedule_id == canonical_fingerprint(
        {**payload, "plc_search_budget": [1 << 20, 1 << 28]}
    )
    rebuilt = restore_model_archive_value(declared)
    assert isinstance(rebuilt, NativeVolumeSchedule)
    assert rebuilt.plc_search_budget == PlcSearchBudget(1 << 20, 1 << 28)
    assert rebuilt.schedule_id == declared.schedule_id
    for invalid in (PlcSearchBudget(0, 1), PlcSearchBudget(1, 1 << 63), (1, 1)):
        with pytest.raises(ValueError, match="plc_search_budget"):
            NativeVolumeSchedule(plc_search_budget=cast(Any, invalid))


def test_soft_shape_strength_is_refused_where_shape_rows_are_not_published() -> None:
    from phydrax.meshing._polyhedral_generation import (
        generate_polyhedral_volume,
        NativePolyhedralSchedule,
    )
    from phydrax.meshing.providers._native_dual import (
        balanced_hex_support_issues,
        dual_hex_support_issues,
        frame_hex_support_issues,
        hex_dominant_support_issues,
    )

    source = _source()
    specification = _request(source)
    soft = NativeVolumeSchedule(shape_strength=SizeControlStrength.SOFT)
    for route, support_issues in (
        ("plc_dual_hex", dual_hex_support_issues),
        ("plc_hex_dominant", hex_dominant_support_issues),
        ("plc_balanced_grid_hex", balanced_hex_support_issues),
        ("plc_frame_grid_hex", frame_hex_support_issues),
    ):
        issue = "a hard-strength native tetrahedral scaffold schedule"
        options = NativeMeshingOptions(route, volume_schedule=soft)
        assert issue in support_issues(source, specification, options)
        hard_options = NativeMeshingOptions(route, volume_schedule=NativeVolumeSchedule())
        assert issue not in support_issues(source, specification, hard_options)
    with pytest.raises(ValueError, match="hard-strength volume schedule"):
        generate_polyhedral_volume(
            source,
            specification,
            NativePolyhedralSchedule(),
            frame_id=SpatialCoordinateContract.si().spatial_id,
            volume_schedule=soft,
        )


def _construct(
    source: NativePlcSource, specification: VolumeMeshingSpec
) -> VolumeConstruction:
    return generate_plc_volume(
        source.complex,
        specification,
        _schedule(),
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_map_id="",
        frame_id=SpatialCoordinateContract.si().spatial_id,
        source_domain=source.mapped_domain,
    )


def test_fill_size_convention_reads_hard_size_and_separates_fill_identity() -> None:
    source = _grouped_cube_source("conforming", 0.0)
    base = _request(source, region_seeds=_CUBE_SEED)
    size = 0.12
    specification = VolumeMeshingSpec(
        base.target,
        base.boundary_scope,
        base.fill_strategy,
        size_controls=(
            UniformSizeControl(
                base.boundary_scope,
                size,
                maximum_size=1.5 * size,
                strength=SizeControlStrength.HARD,
            ),
        ),
        region_seeds=base.region_seeds,
    )

    def construct(**convention: Any) -> VolumeConstruction:
        return generate_plc_volume(
            source.complex,
            specification,
            NativeVolumeSchedule(radius_edge_bound=2.0, minimum_dihedral_degrees=0.0),
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_map_id="",
            frame_id=SpatialCoordinateContract.si().spatial_id,
            **convention,
        )

    default = construct()
    edge = construct(size_convention="edge-length")
    circumradius = construct(size_convention="circumradius")
    # The default is the edge-length convention, bit for bit and by identity.
    assert default.size_convention == edge.size_convention == "edge-length"
    np.testing.assert_array_equal(
        np.asarray(edge.mesh.coordinates).view(np.uint64),
        np.asarray(default.mesh.coordinates).view(np.uint64),
    )
    np.testing.assert_array_equal(
        edge.mesh.blocks[0].vertices, default.mesh.blocks[0].vertices
    )
    assert [stage.report_id for stage in edge.stages] == [
        stage.report_id for stage in default.stages
    ]
    assert default.stages[0].input_ids == (source.complex.complex_id,)
    # Circumradius h refines less than circumradius sqrt(6)/4 h, and the
    # convention joins the fill stage identity.
    assert circumradius.size_convention == "circumradius"
    assert circumradius.stages[0].input_ids[0] == source.complex.complex_id
    assert len(circumradius.stages[0].input_ids) == 2
    assert circumradius.stages[0].report_id != default.stages[0].report_id
    assert (
        circumradius.mesh.blocks[0].vertices.shape[0]
        < default.mesh.blocks[0].vertices.shape[0]
    )
    with pytest.raises(ValueError, match="size_convention"):
        construct(size_convention="edge")


def test_original_plc_curve_witness_rejects_missing_chain_and_wrong_scope(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    from phydrax.geometry._source_fidelity import validate_source_entity_scope
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, write_model_artifact
    from phydrax.meshing.providers._native_plc_geometry import (
        physical_plc_tetrahedra,
        plc_curve_chart_witness,
    )
    from tools.meshing_qualification.scenarios import constrained_solid_matrix as scenario

    profile = next(
        item for item in scenario._PROFILES if item.profile_id == "plc-material-cavity"
    )
    source_input = scenario._input_for(profile)
    source = scenario._source_for(profile, source_input)
    specification, options, _ = scenario._request(profile, source, source_input)
    schedule = options.volume_schedule
    assert schedule is not None
    construction = generate_plc_volume(
        source.complex,
        specification,
        schedule,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_map_id="",
        frame_id=SpatialCoordinateContract.si().spatial_id,
    )
    geometry = physical_plc_tetrahedra(source, construction).geometry
    selected = np.asarray(
        specification.protected_features[0].scope.entity_ids, dtype=np.int64
    )
    query, witness = plc_curve_chart_witness(source, construction, geometry, selected)
    query.validate_chart_witness(witness)
    validate_source_entity_scope(query, source, 1, selected)
    other = next(
        int(value) for value in source.scope_entity_ids(1) if value not in selected
    )
    with pytest.raises(ValueError, match="omits or adds"):
        validate_source_entity_scope(
            query, source, 1, np.asarray((other,), dtype=np.int64)
        )
    source_rows = np.asarray(construction.source_segment_complex_rows)
    segment_rows = np.asarray(construction.segment_source_edges)
    selected_rows = np.flatnonzero(
        np.asarray(source.complex.plc_edge_entity_ids)[source_rows[segment_rows]]
        == selected[0]
    )
    assert selected_rows.size
    keep = np.ones(segment_rows.size, dtype=np.bool_)
    keep[selected_rows[0]] = False
    incomplete = replace(
        construction,
        segments=np.asarray(construction.segments)[keep],
        segment_source_edges=segment_rows[keep],
    )
    with pytest.raises(ValueError, match="gap or overlap"):
        plc_curve_chart_witness(source, incomplete, geometry, selected)
    entry = CompositionEntry(
        query,
        entry_id="plc/curve",
        role="model-state",
        owner_id=source.source_id,
        structure_id=query.source_query_id,
        revision_id=source.source_revision,
        semantics_id="exact-plc-curve",
    )
    archive = write_model_artifact(
        tmp_path / "curve.phx",
        Composition((entry,), boundary_id="plc-curve-roundtrip"),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    script = (
        "import sys; from phydrax.lifecycle import Composition; "
        "from phydrax.geometry._source_fidelity import validate_source_entity_scope; "
        "from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS,read_model_artifact,restore_model_archive_value; "
        "c=read_model_artifact(sys.argv[1],expected_type=Composition,restore=restore_model_archive_value,limits=MODEL_ARCHIVE_LIMITS); "
        "q=c.value('plc/curve'); q.require_current(); "
        "assert q.source_query_id==sys.argv[2]; "
        "validate_source_entity_scope(q,q.owner.source,1,q.entity_ids)"
    )
    subprocess.run(
        [sys.executable, "-c", script, str(archive), query.source_query_id],
        check=True,
    )


def test_original_larger_cavity_splits_embedded_curve_to_meet_quality() -> None:
    # Native refinement used to leave the embedded hard curve as one edge of
    # length sqrt(3)/2 at the finer authored target, so every cell around it
    # was refused and three radius-edge residuals remained.
    from phydrax.meshing.providers._native_plc_geometry import (
        physical_plc_tetrahedra,
        plc_curve_chart_witness,
    )
    from tools.meshing_qualification.scenarios import constrained_solid_matrix as scenario

    profile = next(
        item
        for item in scenario._PROFILES
        if item.profile_id == "plc-material-cavity:larger"
    )
    source_input = scenario._input_for(profile)
    source = scenario._source_for(profile, source_input)
    specification, options, _ = scenario._request(profile, source, source_input)
    schedule = options.volume_schedule
    assert schedule is not None
    construction = generate_plc_volume(
        source.complex,
        specification,
        schedule,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_map_id="",
        frame_id=SpatialCoordinateContract.si().spatial_id,
    )
    assert construction.final_quality_schedule_unmet == ()
    selected = np.asarray(
        specification.protected_features[0].scope.entity_ids, dtype=np.int64
    )
    source_rows = np.asarray(construction.source_segment_complex_rows)
    segment_rows = np.asarray(construction.segment_source_edges)
    curve_rows = np.flatnonzero(
        np.asarray(source.complex.plc_edge_entity_ids)[source_rows[segment_rows]]
        == selected[0]
    )
    assert curve_rows.size > 1
    geometry = physical_plc_tetrahedra(source, construction).geometry
    query, witness = plc_curve_chart_witness(source, construction, geometry, selected)
    query.validate_chart_witness(witness)


def test_final_native_evidence_controls_volume_compliance() -> None:
    source = _source()
    specification = _request(source)
    construction = _construct(source, specification)
    points = np.asarray(construction.mesh.coordinates)

    def issues(candidate: VolumeConstruction) -> tuple[str, ...]:
        return _volume_compliance(
            source, specification, _schedule(), candidate, points
        ).issues

    baseline = replace(
        construction,
        refinement_status=MeshcoreStatus.UNRESOLVED,
        refinement_unmet=(("1", "1", 7, 3.0),),
        improvement_status=MeshcoreStatus.UNRESOLVED,
        improvement_unmet=(("1", "4", 8, 3.0),),
        final_quality_unmet=(),
    )
    assert not issues(baseline)
    assert "native_final_quality:2" in issues(
        replace(baseline, final_quality_unmet=(("2", "4", 8, 1.1),))
    )
    size_only = replace(baseline, final_quality_unmet=(("1", "4", 9, 3.0),))
    assert not issues(size_only)
    assert size_only.final_quality_unmet
    assert "native_final_quality:0" in issues(
        replace(baseline, final_quality_unmet=(("0", "4", 9, 3.0),))
    )


def test_native_size_preference_does_not_override_hard_compliance() -> None:
    source = _source()
    base = _request(source)
    soft = UniformSizeControl(base.boundary_scope, 0.1, strength=SizeControlStrength.SOFT)
    hard_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2)[:1],
    )
    hard = UniformSizeControl(
        hard_scope, 1.5, strength=SizeControlStrength.HARD, priority=1
    )

    def request(hard_control: UniformSizeControl) -> VolumeMeshingSpec:
        return VolumeMeshingSpec(
            base.target,
            base.boundary_scope,
            base.fill_strategy,
            size_controls=(soft, hard_control),
            region_seeds=base.region_seeds,
            size_combination=SizeCombinationPolicy.EXPLICIT_PRIORITY,
            size_compliance=SizeCompliancePolicy(absolute_tolerance=0.6),
        )

    specification = request(hard)
    construction = _construct(source, specification)
    assert any(row[0] == "1" for row in construction.final_quality_unmet)
    assert not construction.final_quality_schedule_unmet
    points = np.asarray(construction.mesh.coordinates)
    report = _volume_compliance(source, specification, _schedule(), construction, points)
    assert not report.issues
    assert dict(report.achieved)["final_quality_unmet_count"] > 0

    failing_hard = UniformSizeControl(
        hard_scope, 0.1, strength=SizeControlStrength.HARD, priority=1
    )
    failure = _volume_compliance(
        source, request(failing_hard), _schedule(), construction, points
    )
    assert any(
        issue.startswith(f"size:{failing_hard.control_id}:") for issue in failure.issues
    )


def test_small_angle_wedge_exempts_only_input_forced_rows() -> None:
    # The authored wedge apex ridge meets at 2 atan(1/16) = 7.153 degrees, below
    # the 10-degree aim, so no tetrahedron on that ridge can meet the aim there.
    # The triangular caps have the same apex corner angle, so every cap triangle
    # at the apex has radius-edge ratio at least 1/(2 sin alpha) = 257/64 > 2.
    from phydrax.meshing._input_forced_quality import (
        input_forced_dihedral_exemptions,
        input_forced_radius_edge_exemptions,
    )
    from tools.meshing_qualification.scenarios import constrained_solid_matrix as scenario

    profile = next(
        item for item in scenario._PROFILES if item.profile_id == "plc-small-angle"
    )
    source_input = scenario._input_for(profile)
    source = scenario._source_for(profile, source_input)
    specification, options, _ = scenario._request(profile, source, source_input)
    declared = options.volume_schedule
    assert declared is not None and declared.minimum_dihedral_degrees == 10.0
    assert declared.radius_edge_bound == 2.0
    # The scenario declares these shape targets soft; the input-forced
    # exemptions are the acceptance rule of the same targets declared hard.
    schedule = NativeVolumeSchedule(
        radius_edge_bound=declared.radius_edge_bound,
        minimum_dihedral_degrees=declared.minimum_dihedral_degrees,
    )
    assert schedule.shape_strength is SizeControlStrength.HARD
    construction = generate_plc_volume(
        source.complex,
        specification,
        schedule,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_map_id="",
        frame_id=SpatialCoordinateContract.si().spatial_id,
    )
    points = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    complex_ = source.complex
    apex = next(
        row
        for row, edge in enumerate(complex_.plc_edge_vertices)
        if sorted(edge) == [0, 3]
    )
    apex_entity = int(complex_.plc_edge_entity_ids[apex])
    input_angle = math.degrees(2.0 * math.atan(1.0 / 16.0))
    rows = construction.final_quality_schedule_unmet
    exemptions = input_forced_dihedral_exemptions(
        complex_, construction, points, schedule.minimum_dihedral_degrees, rows
    )
    assert len(exemptions) == 2
    cells = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
    cell_ids = np.asarray(construction.mesh.blocks[0].global_ids, dtype=np.int64)
    on_apex = np.all(points[:, :2] == 0.0, axis=1)
    for exemption in exemptions:
        assert rows[exemption.row][0] == "2"
        assert exemption.achieved_degrees == rows[exemption.row][3]
        assert exemption.achieved_degrees == pytest.approx(3.576, abs=5e-4)
        assert len(exemption.ridges) == 1
        ridge = exemption.ridges[0]
        assert (ridge.entity_id, ridge.ridge_id) == (
            apex_entity,
            complex_.plc_edge_ids[apex],
        )
        assert ridge.input_angle_degrees == pytest.approx(input_angle, abs=1e-12)
        (cell_row,) = np.flatnonzero(cell_ids == exemption.cell_id)
        assert np.count_nonzero(on_apex[cells[cell_row]]) == 2

    def has_apex_cap_face(cell_row: int) -> bool:
        cell = cells[cell_row]
        for omitted in range(4):
            face = np.delete(cell, omitted)
            heights = points[face, 2]
            if (
                np.all(heights == heights[0])
                and heights[0] in (0.0, 1.0)
                and np.any(on_apex[face])
            ):
                return True
        return False

    native_rows = {
        int(identifier): row
        for row, identifier in enumerate(construction.native_cell_ids.tolist())
    }
    radius_exemptions = input_forced_radius_edge_exemptions(
        complex_, construction, points, schedule.radius_edge_bound, rows
    )
    expected_radius = {
        index
        for index, row in enumerate(rows)
        if row[0] == "0" and has_apex_cap_face(native_rows[row[2]])
    }
    assert expected_radius
    assert {exemption.row for exemption in radius_exemptions} == expected_radius
    apex_vertices = {
        int(complex_.vertex_entity_ids[0]): ((0, 1), (0, 2)),
        int(complex_.vertex_entity_ids[3]): ((3, 4), (3, 5)),
    }
    edge_entities = {
        tuple(sorted(edge)): int(entity)
        for edge, entity in zip(
            complex_.plc_edge_vertices, complex_.plc_edge_entity_ids, strict=True
        )
    }
    for exemption in radius_exemptions:
        assert exemption.achieved_ratio == rows[exemption.row][3]
        assert len(exemption.corners) == 1
        corner = exemption.corners[0]
        assert corner.edge_entity_ids == tuple(
            sorted(edge_entities[edge] for edge in apex_vertices[corner.vertex_entity_id])
        )
        assert corner.input_angle_degrees == pytest.approx(input_angle, abs=1e-12)
        assert corner.forced_ratio == 257 / 64
        assert corner.face_ratio >= corner.forced_ratio * (1.0 - 1e-12)
        assert exemption.achieved_ratio >= corner.face_ratio * (1.0 - 1e-9)

    report = _volume_compliance(source, specification, schedule, construction, points)
    exempt_rows = {exemption.row for exemption in exemptions} | expected_radius
    assert report.issues == tuple(
        f"native_final_quality:{row[0]}"
        for index, row in enumerate(rows)
        if index not in exempt_rows
    )
    achieved = dict(report.achieved)
    assert achieved["input_forced_dihedral_exemption_count"] == 2.0
    assert achieved[
        f"input_forced_dihedral:ridge:{apex_entity}:input_angle_degrees"
    ] == pytest.approx(input_angle, abs=1e-12)
    for exemption in exemptions:
        key = (
            f"input_forced_dihedral:cell:{exemption.cell_id}:ridge:{apex_entity}"
            ":achieved_degrees"
        )
        assert achieved[key] == exemption.achieved_degrees
    assert achieved["input_forced_radius_edge_exemption_count"] == len(expected_radius)
    for exemption in radius_exemptions:
        corner = exemption.corners[0]
        corner_key = (
            f"vertex:{corner.vertex_entity_id}:edges:"
            f"{corner.edge_entity_ids[0]}:{corner.edge_entity_ids[1]}"
        )
        prefix = f"input_forced_radius_edge:cell:{exemption.cell_id}:{corner_key}"
        assert achieved[f"{prefix}:face_ratio"] == corner.face_ratio
        assert achieved[f"{prefix}:achieved_ratio"] == exemption.achieved_ratio
        assert achieved[
            f"input_forced_radius_edge:{corner_key}:input_angle_degrees"
        ] == pytest.approx(input_angle, abs=1e-12)
        assert achieved[f"input_forced_radius_edge:{corner_key}:forced_ratio"] == 257 / 64

    def issues(
        quality_rows: tuple[tuple[str, str, int, float], ...],
        minimum_dihedral_degrees: float = 10.0,
        radius_edge_bound: float = 2.0,
    ) -> tuple[str, ...]:
        return _volume_compliance(
            source,
            specification,
            NativeVolumeSchedule(
                radius_edge_bound=radius_edge_bound,
                minimum_dihedral_degrees=minimum_dihedral_degrees,
            ),
            replace(construction, final_quality_unmet=quality_rows),
            points,
        ).issues

    forced_row = exemptions[0].row
    forced_native = rows[forced_row][2]
    # A radius-edge row on the dihedral-forced cell is exempt only for a cap face.
    assert issues((("0", "4", forced_native, 3.4), rows[forced_row])) == (
        ()
        if has_apex_cap_face(native_rows[forced_native])
        else ("native_final_quality:0",)
    )
    # The same dihedral is hard when the ridge angle meets the requested aim.
    assert issues((rows[forced_row],), 5.0) == ("native_final_quality:2",)
    # A small dihedral in a cell off the sub-target ridge stays hard.
    off_ridge = next(
        row for row in range(cells.shape[0]) if np.count_nonzero(on_apex[cells[row]]) < 2
    )
    off_native = int(construction.native_cell_ids[off_ridge])
    assert issues((("2", "4", off_native, 3.576),)) == ("native_final_quality:2",)
    # A radius-edge row on a cell without a forcing cap face stays hard.
    off_cap = next(row for row in range(cells.shape[0]) if not has_apex_cap_face(row))
    off_cap_native = int(construction.native_cell_ids[off_cap])
    assert issues((("0", "4", off_cap_native, 3.4),)) == ("native_final_quality:0",)
    # A forced cap row is hard when 1/(2 sin alpha) does not exceed the bound.
    cap_row = rows[min(expected_radius)]
    assert issues((cap_row,), radius_edge_bound=5.0) == ("native_final_quality:0",)


def test_small_angle_wedge_soft_shape_targets_publish_unmet_rows() -> None:
    from phydrax._meshcore import TetMeshUnmetReason
    from tools.meshing_qualification.scenarios import constrained_solid_matrix as scenario

    profile = next(
        item for item in scenario._PROFILES if item.profile_id == "plc-small-angle"
    )
    source_input = scenario._input_for(profile)
    source = scenario._source_for(profile, source_input)
    specification, options, _ = scenario._request(profile, source, source_input)
    soft = options.volume_schedule
    assert soft is not None and soft.shape_strength is SizeControlStrength.SOFT
    hard = NativeVolumeSchedule(
        radius_edge_bound=soft.radius_edge_bound,
        minimum_dihedral_degrees=soft.minimum_dihedral_degrees,
    )
    assert hard.shape_strength is SizeControlStrength.HARD

    def construct(schedule: NativeVolumeSchedule) -> VolumeConstruction:
        return generate_plc_volume(
            source.complex,
            specification,
            schedule,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_map_id="",
            frame_id=SpatialCoordinateContract.si().spatial_id,
        )

    construction = construct(soft)
    # Strength is acceptance-side: native refinement targets are identical.
    hard_construction = construct(hard)
    assert construction.mesh.topology_id == hard_construction.mesh.topology_id
    assert construction.final_quality_unmet == hard_construction.final_quality_unmet
    points = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    rows = construction.final_quality_schedule_unmet
    shape_rows = tuple(row for row in rows if row[0] in ("0", "2"))
    assert shape_rows

    hard_report = _volume_compliance(source, specification, hard, construction, points)
    explicit_hard = _volume_compliance(
        source,
        specification,
        NativeVolumeSchedule(
            radius_edge_bound=hard.radius_edge_bound,
            minimum_dihedral_degrees=hard.minimum_dihedral_degrees,
            shape_strength=SizeControlStrength.HARD,
        ),
        construction,
        points,
    )
    assert (explicit_hard.issues, explicit_hard.requested, explicit_hard.achieved) == (
        hard_report.issues,
        hard_report.requested,
        hard_report.achieved,
    )
    hard_achieved = dict(hard_report.achieved)
    assert hard_achieved["input_forced_dihedral_exemption_count"] == 2.0
    assert hard_report.issues
    assert not any(key.startswith("soft_shape_target") for key in hard_achieved)

    report = _volume_compliance(source, specification, soft, construction, points)
    assert report.issues == ()
    assert report.requested == hard_report.requested
    achieved = dict(report.achieved)
    assert achieved["soft_shape_target_unmet_count"] == float(len(shape_rows))
    assert not any(key.startswith("input_forced_") for key in achieved)
    published = dict(
        zip(
            construction.native_cell_ids.tolist(),
            construction.mesh.blocks[0].global_ids.tolist(),
            strict=True,
        )
    )
    names = {"0": "radius_edge_ratio", "2": "dihedral_degrees"}
    for criterion, reason, native_cell, value in shape_rows:
        key = (
            f"soft_shape_target:cell:{published[native_cell]}:{names[criterion]}:"
            f"{TetMeshUnmetReason(int(reason)).name.lower()}"
        )
        assert achieved[key] == value

    def soft_issues(
        quality_rows: tuple[tuple[str, str, int, float], ...],
    ) -> tuple[str, ...]:
        return _volume_compliance(
            source,
            specification,
            soft,
            replace(construction, final_quality_unmet=quality_rows),
            points,
        ).issues

    cell = shape_rows[0][2]
    # Validity, determinant and nonrepresentable rows stay hard under soft.
    assert soft_issues((("4", "4", cell, -0.1),)) == ("native_final_quality:4",)
    assert soft_issues((("4", "2", cell, 0.0),)) == ("native_final_quality:4",)
    assert soft_issues((("3", "2", cell, math.nan),)) == ("native_final_quality:3",)
    assert soft_issues((("0", "2", cell, math.inf),)) == ("native_final_quality:0",)

    plan = NativeMeshingProvider(
        NativeMeshingOptions("plc_tetrahedral", volume_schedule=soft)
    ).plan(source, specification, coordinate_contract=scenario._CONTRACT)
    result = plan.execute()
    assert result.compliance.issues == ()
    result_achieved = dict(result.compliance.achieved)
    assert result_achieved["soft_shape_target_unmet_count"] > 0

    # The implicit adaptive route's own aggregate shape gates follow the strength.
    from phydrax.meshing.providers._native_implicit import _implicit_volume_compliance

    shape_issues = {"radius_edge_bound", "minimum_dihedral_degrees"}
    implicit_hard = _implicit_volume_compliance(
        specification, hard, construction, 0.0, ()
    )
    assert shape_issues <= set(implicit_hard.issues)
    assert "native_final_quality:0" in implicit_hard.issues
    implicit_soft = _implicit_volume_compliance(
        specification, soft, construction, 0.0, ()
    )
    assert not shape_issues & set(implicit_soft.issues)
    assert not any(
        issue.startswith("native_final_quality") for issue in implicit_soft.issues
    )
    assert dict(implicit_soft.achieved)["soft_shape_target_unmet_count"] == float(
        len(shape_rows)
    )


def _prepare(
    source: NativePlcSource, specification: VolumeMeshingSpec
) -> NativeMeshConstruction:
    options = NativeMeshingOptions("plc_tetrahedral", volume_schedule=_schedule())
    return prepare_native_plc_volume(
        source,
        specification,
        options,
        SpatialCoordinateContract.si(),
    ).execute()


def test_plc_ids_and_occurrences_are_authored_and_not_coordinate_derived() -> None:
    constructor = cast(Callable[..., PiecewiseLinearComplex], PiecewiseLinearComplex)
    with pytest.raises(TypeError, match="vertex_occurrence_paths"):
        constructor(
            _POINTS,
            _POLYGONS,
            np.arange(_POLYGONS.shape[0], dtype=np.int64),
            _FACET_REGIONS,
            ("upper", "lower"),
            segments=np.empty((0, 2), dtype=np.int64),
            vertex_ids=tuple(f"vertex-{row}" for row in range(_POINTS.shape[0])),
            facet_ids=tuple(f"facet-{row}" for row in range(_POLYGONS.shape[0])),
            facet_occurrence_paths=((),) * _POLYGONS.shape[0],
            segment_ids=(),
            segment_occurrence_paths=(),
            region_occurrence_paths=((), ()),
        )

    source = _source()
    moved = np.asarray(_POINTS, dtype=np.float64).copy()
    moved[3, 2] = 1.25
    changed_geometry = PiecewiseLinearComplex(
        moved,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=source.complex.vertex_ids,
        vertex_occurrence_paths=source.complex.vertex_occurrence_paths,
        facet_ids=source.complex.facet_ids,
        facet_occurrence_paths=source.complex.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=source.complex.region_occurrence_paths,
    )
    assert changed_geometry.vertex_ids == source.complex.vertex_ids
    assert changed_geometry.vertex_entity_ids == source.complex.vertex_entity_ids
    assert changed_geometry.complex_id != source.complex.complex_id

    permutation = np.asarray((2, 0, 4, 1, 3), dtype=np.int64)
    old_to_new = np.empty_like(permutation)
    old_to_new[permutation] = np.arange(permutation.size, dtype=np.int64)
    reordered = PiecewiseLinearComplex(
        _POINTS[permutation],
        old_to_new[_POLYGONS],
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(source.complex.vertex_ids[int(row)] for row in permutation),
        vertex_occurrence_paths=tuple(
            source.complex.vertex_occurrence_paths[int(row)] for row in permutation
        ),
        facet_ids=source.complex.facet_ids,
        facet_occurrence_paths=source.complex.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=source.complex.region_occurrence_paths,
    )
    original_vertex_ids = dict(
        zip(source.complex.vertex_ids, source.complex.vertex_entity_ids, strict=True)
    )
    reordered_vertex_ids = dict(
        zip(reordered.vertex_ids, reordered.vertex_entity_ids, strict=True)
    )
    assert original_vertex_ids == reordered_vertex_ids
    original_edge_ids = dict(
        zip(source.complex.plc_edge_ids, source.complex.plc_edge_entity_ids, strict=True)
    )
    reordered_edge_ids = dict(
        zip(reordered.plc_edge_ids, reordered.plc_edge_entity_ids, strict=True)
    )
    assert original_edge_ids == reordered_edge_ids

    repeated = PiecewiseLinearComplex(
        _POINTS,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=("shared", "shared", "vertex-2", "vertex-3", "vertex-4"),
        vertex_occurrence_paths=(
            ("assembly-a",),
            ("assembly-b",),
            (),
            (),
            (),
        ),
        facet_ids=source.complex.facet_ids,
        facet_occurrence_paths=source.complex.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=source.complex.region_occurrence_paths,
    )
    assert repeated.vertex_entity_ids[0] != repeated.vertex_entity_ids[1]
    repeated_face_ids = _source_face_ids(
        np.asarray(((0, 2, 3), (1, 2, 3)), dtype=np.int64),
        np.zeros((2,), dtype=np.int64),
        repeated.vertex_ids,
        repeated.vertex_occurrence_paths,
        repeated.facet_ids,
        repeated.facet_occurrence_paths,
    )
    assert len(set(repeated_face_ids)) == 2


def test_shared_polygon_loop_seam_is_a_stable_plc_edge() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    loops = (
        np.asarray((0, 1, 2), dtype=np.int64),
        np.asarray((0, 2, 3), dtype=np.int64),
    )

    def make_complex(polygons: tuple[np.ndarray, ...], /) -> PiecewiseLinearComplex:
        return PiecewiseLinearComplex(
            points,
            polygons,
            np.zeros((2,), dtype=np.int64),
            np.asarray(((-1, 0),), dtype=np.int64),
            ("solid",),
            segments=np.empty((0, 2), dtype=np.int64),
            vertex_ids=("v0", "v1", "v2", "v3"),
            vertex_occurrence_paths=((),) * 4,
            facet_ids=("one-quad-facet",),
            facet_occurrence_paths=((),),
            segment_ids=(),
            segment_occurrence_paths=(),
            region_occurrence_paths=((),),
        )

    complex_ = make_complex(loops)
    reordered = make_complex(tuple(reversed(loops)))
    rows = {
        tuple(sorted(edge)): row for row, edge in enumerate(complex_.plc_edge_vertices)
    }
    assert set(rows) == {(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)}
    seam = rows[(0, 2)]
    assert complex_.plc_edge_facets[seam] == (0,)
    reordered_rows = {
        tuple(sorted(edge)): row for row, edge in enumerate(reordered.plc_edge_vertices)
    }
    assert complex_.plc_edge_ids[seam] == reordered.plc_edge_ids[reordered_rows[(0, 2)]]
    assert (
        complex_.plc_edge_entity_ids[seam]
        == reordered.plc_edge_entity_ids[reordered_rows[(0, 2)]]
    )


def test_grouped_coplanar_facet_diagonals_are_facet_owned_unless_requested() -> None:
    source = _grouped_cube_source()
    seed = RegionSeed(
        np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
        "solid",
        "solid-material",
        RegionRole.SOLID,
    )
    construction = _construct(source, _request(source, region_seeds=(seed,)))
    complex_ = source.complex
    boundary_rows = {
        row for row, facets in enumerate(complex_.plc_edge_facets) if len(facets) == 2
    }
    diagonal_rows = set(range(len(complex_.plc_edge_ids))) - boundary_rows
    assert len(boundary_rows) == 12
    assert len(diagonal_rows) == 6
    boundary_ids = {complex_.plc_edge_ids[row] for row in boundary_rows}
    diagonal_ids = {complex_.plc_edge_ids[row] for row in diagonal_rows}
    assert set(construction.segment_source_ids) == boundary_ids
    assert not set(construction.segment_source_ids) & diagonal_ids
    assert set(construction.face_source_ids) == set(complex_.facet_ids)
    assert construction.region_ids == ("solid",)
    np.testing.assert_array_equal(
        construction.cell_regions, np.zeros_like(construction.cell_regions)
    )

    requested_diagonal_row = min(diagonal_rows)
    feature_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        1,
        source.scope_entity_set_id(1),
        np.asarray(
            (complex_.plc_edge_entity_ids[requested_diagonal_row],), dtype=np.int64
        ),
    )
    requested_feature = ProtectedFeature(feature_scope, FeatureKind.CURVE)
    base_specification = _request(source, region_seeds=(seed,))
    feature_specification = VolumeMeshingSpec(
        base_specification.target,
        base_specification.boundary_scope,
        base_specification.fill_strategy,
        size_controls=base_specification.size_controls,
        protected_features=(requested_feature,),
        region_seeds=base_specification.region_seeds,
    )
    # A requested diagonal is an edge of the authored face triangulation, so
    # even a fixed boundary recovers it without boundary Steiner points.
    requested = _construct(source, feature_specification)
    requested_id = complex_.plc_edge_ids[requested_diagonal_row]
    assert set(requested.segment_source_ids) == boundary_ids | {requested_id}
    assert set(requested.face_source_ids) == set(complex_.facet_ids)
    assert np.all(np.asarray(requested.boundary_faces) < complex_.vertices.shape[0])


def _declared_rotated_cube(
    facet_planes: np.ndarray | None,
    facet_plane_tolerances: np.ndarray | None = None,
    boundary: PlcBoundaryPolicy = "fixed",
) -> NativePlcSource:
    base = _grouped_cube_source(angle=0.3).complex
    loops = np.asarray(base.polygon_vertices).reshape(-1, 3)
    complex_ = PiecewiseLinearComplex(
        base.vertices,
        loops,
        base.polygon_facets,
        base.facet_regions,
        base.region_ids,
        segments=base.segments,
        vertex_ids=base.vertex_ids,
        vertex_occurrence_paths=base.vertex_occurrence_paths,
        facet_ids=base.facet_ids,
        facet_occurrence_paths=base.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=base.region_occurrence_paths,
        boundary=boundary,
        facet_planes=facet_planes,
        facet_plane_tolerances=facet_plane_tolerances,
    )
    return NativePlcSource(complex_, "declared-cube", "revision-1")


def _rotated_cube_planes() -> np.ndarray:
    points = np.asarray(_grouped_cube_source(angle=0.3).complex.vertices)
    corners = ((0, 2, 1), (4, 5, 6), (0, 1, 4), (2, 6, 3), (0, 4, 2), (1, 3, 5))
    return np.asarray(
        [(points[a], points[b] - points[a], points[c] - points[a]) for a, b, c in corners]
    )


def test_declared_facet_planes_group_rounded_cube_faces_without_diagonal_segments() -> (
    None
):
    seed = RegionSeed(
        np.zeros((3,), dtype=np.float64), "solid", "solid-material", RegionRole.SOLID
    )
    undeclared = _declared_rotated_cube(None)
    declared = _declared_rotated_cube(_rotated_cube_planes(), np.full((6,), 1e-12))
    # The rotated faces' two triangles are not exactly coplanar: undeclared,
    # each face diagonal is a PLC edge and a recovered segment source.
    assert undeclared.complex.facet_planes is None
    assert (
        undeclared.complex.complex_id
        == _grouped_cube_source(angle=0.3).complex.complex_id
    )
    assert declared.complex.complex_id != undeclared.complex.complex_id
    loose = _construct(undeclared, _request(undeclared, region_seeds=(seed,)))
    grouped = _construct(declared, _request(declared, region_seeds=(seed,)))
    complex_ = declared.complex
    boundary_ids = {
        complex_.plc_edge_ids[row]
        for row, facets in enumerate(complex_.plc_edge_facets)
        if len(facets) == 2
    }
    assert len(boundary_ids) == 12
    assert len(set(loose.segment_source_ids)) == 18
    assert set(grouped.segment_source_ids) == boundary_ids
    assert set(grouped.face_source_ids) == set(complex_.facet_ids)


def test_declared_facet_planes_are_validated_rows() -> None:
    planes = _rotated_cube_planes()
    with pytest.raises(ValueError, match="facet_plane_tolerances require"):
        _declared_rotated_cube(None, np.zeros((6,)))
    partial = planes.copy()
    partial[2, 1, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        _declared_rotated_cube(partial)
    dependent = planes.copy()
    dependent[2, 2] = 2.0 * dependent[2, 1]
    with pytest.raises(ValueError, match="exactly independent"):
        _declared_rotated_cube(dependent)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _declared_rotated_cube(planes, np.full((6,), -1.0))
    undeclared_rows = np.full((6, 3, 3), np.nan)
    accepted = _declared_rotated_cube(undeclared_rows)
    np.testing.assert_array_equal(accepted.complex.facet_plane_tolerances, np.zeros((6,)))


@pytest.mark.parametrize("floor_plane", ("implicit", "declared", "undeclared"))
def test_curved_facet_keeps_authored_diagonal_of_exactly_coplanar_trapezoid(
    floor_plane: str,
) -> None:
    # A hip roof: the front and back trapezoids are exactly coplanar triangle
    # pairs inside one non-planar "roof" facet, as symmetric trapezoids are on
    # a discrete surface of revolution. Both trapezoids are cyclic, so either
    # diagonal is a valid recovery; the authored one must be kept. The planar
    # floor merges its two triangles without plane declarations or when its
    # plane is declared; a complex declaring planes keeps the diagonal of an
    # undeclared floor.
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (4.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
            (1.0, 1.0, 1.0),
            (3.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    roof = ((0, 1, 5), (0, 5, 4), (2, 3, 4), (2, 4, 5), (1, 2, 5), (3, 0, 4))
    floor = ((0, 3, 2), (0, 2, 1))
    facet_planes = None
    if floor_plane != "implicit":
        facet_planes = np.full((2, 3, 3), np.nan, dtype=np.float64)
        if floor_plane == "declared":
            facet_planes[1] = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    complex_ = PiecewiseLinearComplex(
        points,
        np.asarray(roof + floor, dtype=np.int64),
        np.asarray((0,) * len(roof) + (1,) * len(floor), dtype=np.int64),
        np.asarray(((-1, 0), (-1, 0)), dtype=np.int64),
        ("solid",),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"roof-vertex-{row}" for row in range(points.shape[0])),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=("roof", "floor"),
        facet_occurrence_paths=((), ()),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="fixed",
        facet_planes=facet_planes,
    )
    source = NativePlcSource(complex_, "hip-roof", "revision-1")
    seed = RegionSeed(
        np.asarray((2.0, 1.0, 0.3), dtype=np.float64),
        "solid",
        "solid-material",
        RegionRole.SOLID,
    )
    construction = _construct(source, _request(source, region_seeds=(seed,)))

    faces = np.asarray(construction.input_vertices)[
        np.asarray(construction.boundary_faces)
    ]
    roof_faces = {tuple(sorted(face)) for face in faces.tolist() if {4, 5} & set(face)}
    assert roof_faces == {tuple(sorted(face)) for face in roof}
    edge_ids = {
        tuple(sorted(edge)): identifier
        for edge, identifier in zip(
            complex_.plc_edge_vertices, complex_.plc_edge_ids, strict=True
        )
    }
    segment_ids = set(construction.segment_source_ids)
    assert {edge_ids[(0, 5)], edge_ids[(2, 4)]} <= segment_ids
    floor_diagonal = edge_ids.get((0, 2))
    if floor_plane == "undeclared":
        assert floor_diagonal in segment_ids
    else:
        assert floor_diagonal not in segment_ids


def test_raw_recovery_witnesses_map_to_canonical_source_identity_rows() -> None:
    base = _source("conforming")
    complex_ = PiecewiseLinearComplex(
        base.complex.vertices,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        base.complex.region_ids,
        segments=np.asarray((base.complex.plc_edge_vertices[0],), dtype=np.int64),
        vertex_ids=base.complex.vertex_ids,
        vertex_occurrence_paths=base.complex.vertex_occurrence_paths,
        facet_ids=base.complex.facet_ids,
        facet_occurrence_paths=base.complex.facet_occurrence_paths,
        segment_ids=("explicit-loop-edge",),
        segment_occurrence_paths=(base.complex.plc_edge_occurrence_paths[0],),
        region_occurrence_paths=base.complex.region_occurrence_paths,
        boundary="conforming",
    )
    source = NativePlcSource(complex_, "bipyramid-with-explicit-edge", "revision-1")
    prepared = prepare_plc_source(
        complex_.vertices,
        complex_.polygon_offsets,
        complex_.polygon_vertices,
        complex_.polygon_facets,
        complex_.facet_regions,
        complex_.segments,
    )
    if prepared.status is not MeshcoreStatus.OK:
        raise AssertionError("Native PLC source preparation failed.")
    canonical_edge = tuple(int(value) for value in complex_.plc_edge_vertices[0])
    raw_edges = np.asarray(prepared.plc_edges, dtype=np.int64)
    raw_edge_rows = [
        row
        for row, edge in enumerate(raw_edges.tolist())
        if tuple(sorted(int(value) for value in edge)) == tuple(sorted(canonical_edge))
    ]
    assert len(raw_edge_rows) == 1
    raw_edge_row = raw_edge_rows[0]
    raw_edge = tuple(int(value) for value in raw_edges[raw_edge_row])
    segment_point = 0.75 * np.asarray(
        complex_.vertices[raw_edge[0]], dtype=np.float64
    ) + 0.25 * np.asarray(complex_.vertices[raw_edge[1]], dtype=np.float64)
    source_triangle = np.asarray(prepared.input_triangles[0], dtype=np.int64)
    face_point = (
        np.asarray(complex_.vertices[source_triangle[0]], dtype=np.float64)
        + 0.2
        * (
            np.asarray(complex_.vertices[source_triangle[1]], dtype=np.float64)
            - np.asarray(complex_.vertices[source_triangle[0]], dtype=np.float64)
        )
        + 0.3
        * (
            np.asarray(complex_.vertices[source_triangle[2]], dtype=np.float64)
            - np.asarray(complex_.vertices[source_triangle[0]], dtype=np.float64)
        )
    )
    recovery = PlcRecovery3D(
        prepared.status,
        np.vstack((prepared.points, segment_point, face_point)),
        prepared.tetrahedra,
        prepared.tetrahedron_regions,
        prepared.faces,
        prepared.face_sources,
        prepared.face_orientations,
        prepared.segments,
        prepared.segment_sources,
        np.concatenate((prepared.protection, np.zeros((2,), dtype=np.float64))),
        np.concatenate((prepared.vertex_dimensions, np.asarray((1, 2), dtype=np.int8))),
        prepared.plc_edges,
        prepared.input_triangles,
        prepared.input_polygons,
        np.concatenate((prepared.source_strata, np.asarray((1, 2), dtype=np.int8))),
        np.concatenate(
            (
                prepared.source_entities,
                np.asarray((raw_edge_row, 0), dtype=np.int32),
            )
        ),
        np.concatenate(
            (
                prepared.source_parameters,
                np.asarray(((0.25, 0.0), (0.2, 0.3)), dtype=np.float64),
            )
        ),
        np.concatenate((prepared.source_deviations, np.zeros((2,), dtype=np.float64))),
        prepared.suppression_search,
    )
    canonical_edges = np.asarray(complex_.plc_edge_vertices, dtype=np.int64)
    vertex_identity_rows = tuple(
        zip(complex_.vertex_ids, complex_.vertex_occurrence_paths, strict=True)
    )
    canonical_row_by_endpoints = {
        tuple(sorted(vertex_identity_rows[int(vertex)] for vertex in edge)): row
        for row, edge in enumerate(canonical_edges.tolist())
    }
    raw_edge_complex_rows = np.asarray(
        [
            canonical_row_by_endpoints[
                tuple(sorted(vertex_identity_rows[int(vertex)] for vertex in edge))
            ]
            for edge in recovery.plc_edges.tolist()
        ],
        dtype=np.int64,
    )
    expected_rows = np.unique(raw_edge_complex_rows)
    expected_local_row_by_complex = {
        int(complex_row): local_row
        for local_row, complex_row in enumerate(expected_rows.tolist())
    }
    expected_carrier_source_rows = np.asarray(
        [
            expected_local_row_by_complex[int(complex_row)]
            for complex_row in raw_edge_complex_rows.tolist()
        ],
        dtype=np.int64,
    )
    tables = _source_tables(complex_, _request(source), recovery)
    np.testing.assert_array_equal(tables.source_segment_complex_rows, expected_rows)
    np.testing.assert_array_equal(
        tables.source_segment_vertices, canonical_edges[expected_rows]
    )
    np.testing.assert_array_equal(
        tables.plc_edge_source_rows, expected_carrier_source_rows
    )
    assert tables.segment_entity_ids == tuple(
        complex_.plc_edge_ids[int(row)] for row in expected_rows
    )
    assert tables.segment_occurrence_paths == tuple(
        complex_.plc_edge_occurrence_paths[int(row)] for row in expected_rows
    )

    expected_segment_parameter = 0.25 if raw_edge == canonical_edge else 0.75
    assert int(tables.source.witness_entities[-2]) == 0
    assert tables.source.witness_parameters[-2, 0] == pytest.approx(
        expected_segment_parameter
    )
    assert int(tables.source.witness_entities[-1]) == 0
    np.testing.assert_array_equal(
        tables.source.witness_parameters[-1],
        np.asarray((0.2, 0.3), dtype=np.float64),
    )
    assert tables.segment_entity_ids[0] == complex_.plc_edge_ids[0]
    assert tables.segment_occurrence_paths[0] == complex_.plc_edge_occurrence_paths[0]


def test_native_plc_fill_recovers_material_interface_and_fixed_boundary() -> None:
    source = _source("fixed")
    construction = _construct(source, _request(source))
    assert construction.boundary_policy == "fixed"
    assert isinstance(construction.domain, PiecewiseLinearDomain)
    assert construction.region_ids == ("upper", "lower")
    assert (
        construction.domain.region_occurrence_paths
        == source.complex.region_occurrence_paths
    )
    assert set(np.asarray(construction.cell_regions, dtype=np.int64).tolist()) == {0, 1}
    assert construction.input_vertices.shape == (construction.mesh.coordinates.shape[0],)
    boundary_vertex_rows = np.unique(construction.boundary_faces)
    assert np.all(construction.input_vertices[boundary_vertex_rows] >= 0)
    for face, facet in zip(
        construction.boundary_faces,
        construction.face_source_facets,
        strict=True,
    ):
        source_vertices = construction.input_vertices[np.asarray(face, dtype=np.int64)]
        assert np.all(source_vertices >= 0)
        assert tuple(sorted(source_vertices.tolist())) == tuple(
            sorted(_POLYGONS[int(facet)].tolist())
        )

    interface_facets = np.flatnonzero(np.all(_FACET_REGIONS >= 0, axis=1))
    assert interface_facets.size == 1
    interface_rows = np.flatnonzero(
        np.isin(construction.face_source_facets, interface_facets)
    )
    assert interface_rows.size == 1
    assert (
        construction.face_source_ids[int(interface_rows[0])]
        == source.complex.facet_ids[int(interface_facets[0])]
    )

    result = _prepare(source, _request(source))

    region_evidence = result.organization.region_boundary_evidence
    assert len(region_evidence) == 2
    evidence_by_region = {
        evidence.source_region_id: evidence for evidence in region_evidence
    }
    assert set(evidence_by_region) == {"upper", "lower"}
    for region_id, evidence in evidence_by_region.items():
        assert evidence.control_id is None
        assert evidence.material_id == f"{region_id}-material"
        assert evidence.role is RegionRole.SOLID
        assert evidence.source_scope.source_id == source.source_id
        assert evidence.source_scope.source_revision == source.source_revision
        assert evidence.source_scope.entity_set_id == source.scope_entity_set_id(3)
        assert source.source_entity_key_for_scope(
            3, int(evidence.source_scope.entity_ids[0])
        ) == (evidence.source_region_id, evidence.source_occurrence_path)
        assert evidence.boundary_label.label_id in {
            label.label_id for label in result.organization.labels
        }
        evidence.require_current(
            result.mesh, result.organization.labels, result.organization.patches
        )
    interface_patch = next(
        patch
        for patch in result.organization.patches
        if patch.source_adjacent_region_ids == ("upper", "lower")
    )
    assert interface_patch.source_adjacent_region_ids is not None
    assert interface_patch.source_adjacent_region_occurrence_paths is not None
    assert interface_patch.source_adjacent_region_occurrence_paths == ((), ())
    assert dict(evidence_by_region["upper"].patch_sides)[interface_patch.patch_id] == 1
    assert dict(evidence_by_region["lower"].patch_sides)[interface_patch.patch_id] == -1
    face_association = next(
        association
        for association in result.organization.associations
        if association.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
        and association.target_entity_set_id == result.mesh.entity_set(2).entity_set_id
        and any(
            role is GeometrySourceEntityRole.FACET for role in association.source_roles
        )
    )
    assert any(
        positive is not None and negative is not None and positive != negative
        for positive, negative in face_association.source_region_sides
    )
    assert result.certification.cell_regions is not None
    assert set(
        np.asarray(result.certification.cell_regions, dtype=np.int64).tolist()
    ) == {
        0,
        1,
    }


def test_native_plc_region_control_identity_is_preserved_in_boundary_evidence() -> None:
    source = _source()
    region_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        np.asarray((source.complex.region_entity_ids[0],), dtype=np.int64),
    )
    control = RegionControl(
        region_scope,
        "upper",
        "controlled-upper",
        RegionRole.FLUID,
    )

    construction = _prepare(source, _request(source, region_controls=(control,)))

    upper = next(
        evidence
        for evidence in construction.organization.region_boundary_evidence
        if evidence.source_region_id == "upper"
    )
    assert upper.control_id == control.control_id
    assert upper.source_scope.scope_id == control.scope.scope_id
    assert upper.material_id == "controlled-upper"
    assert upper.role is RegionRole.FLUID
    lower = next(
        evidence
        for evidence in construction.organization.region_boundary_evidence
        if evidence.source_region_id == "lower"
    )
    assert lower.control_id is None


def test_native_plc_boundary_evidence_separates_repeated_region_occurrences() -> None:
    source = _repeated_region_source()
    specification = _request(source, region_seeds=())
    volume = _construct(source, specification)
    organization = _organization(source, specification, volume)

    assert isinstance(volume.domain, PiecewiseLinearDomain)
    assert volume.domain.region_ids == source.complex.region_ids
    assert volume.domain.region_occurrence_paths == source.complex.region_occurrence_paths
    _, accepted_regions = _prepare_region_assignments(
        volume.mesh,
        MeshCertificationSchedule("volume_plc"),
        volume.domain,
        volume.cell_regions,
        source.source_id,
    )
    np.testing.assert_array_equal(accepted_regions, volume.cell_regions)
    evidence_by_path = {
        evidence.source_occurrence_path: evidence
        for evidence in organization.region_boundary_evidence
    }
    assert set(evidence_by_path) == {("assembly-a",), ("assembly-b",)}
    assert all(
        evidence.source_region_id == "shared-region" and evidence.control_id is None
        for evidence in evidence_by_path.values()
    )
    assert (
        len({evidence.source_scope.scope_id for evidence in evidence_by_path.values()})
        == 2
    )
    assert (
        len(
            {
                int(evidence.source_scope.entity_ids[0])
                for evidence in evidence_by_path.values()
            }
        )
        == 2
    )
    zone_names = tuple(zone.name for zone in organization.zones)
    assert len(set(zone_names)) == 2
    assert all(name.startswith("shared-region@") for name in zone_names)
    for evidence in evidence_by_path.values():
        assert source.source_entity_key_for_scope(
            3, int(evidence.source_scope.entity_ids[0])
        ) == (evidence.source_region_id, evidence.source_occurrence_path)
        evidence.require_current(volume.mesh, organization.labels, organization.patches)
    accepted_evidence = _validated_source_evidence(
        volume.mesh,
        SpatialCoordinateContract.si(),
        None,
        organization.region_boundary_evidence,
        None,
        organization.labels,
        organization.patches,
    )
    assert {
        (evidence.source_region_id, evidence.source_occurrence_path)
        for evidence in accepted_evidence
    } == {
        ("shared-region", ("assembly-a",)),
        ("shared-region", ("assembly-b",)),
    }
    interface_patch = next(
        patch
        for patch in organization.patches
        if patch.source_adjacent_region_ids == ("shared-region", "shared-region")
    )
    assert interface_patch.source_adjacent_region_ids is not None
    assert interface_patch.source_adjacent_region_occurrence_paths is not None
    assert interface_patch.source_adjacent_region_occurrence_paths == (
        ("assembly-a",),
        ("assembly-b",),
    )
    assert (
        dict(evidence_by_path[("assembly-a",)].patch_sides)[interface_patch.patch_id] == 1
    )
    assert (
        dict(evidence_by_path[("assembly-b",)].patch_sides)[interface_patch.patch_id]
        == -1
    )


def test_conforming_plc_fill_retains_source_ancestry_and_region_coverage() -> None:
    source = _source("conforming")
    construction = _construct(source, _request(source))
    assert construction.boundary_policy == "conforming"
    assert construction.source_id == source.source_id
    assert construction.source_revision == source.source_revision
    assert construction.complex_id == source.complex.complex_id
    assert len(construction.vertex_source_ids) == construction.mesh.coordinates.shape[0]
    assert len(construction.face_source_ids) == construction.boundary_faces.shape[0]
    assert len(construction.segment_source_ids) == construction.segments.shape[0]
    assert construction.region_ids == tuple(construction.domain.region_ids)


def _conforming_refinement(
    source: NativePlcSource, seeds: tuple[RegionSeed, ...], size: float
) -> VolumeConstruction:
    base = _request(source, region_seeds=seeds)
    specification = VolumeMeshingSpec(
        base.target,
        base.boundary_scope,
        base.fill_strategy,
        size_controls=(UniformSizeControl(base.boundary_scope, size),),
        region_seeds=base.region_seeds,
    )
    return generate_plc_volume(
        source.complex,
        specification,
        NativeVolumeSchedule(radius_edge_bound=2.0, minimum_dihedral_degrees=0.0),
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_map_id="",
        frame_id=SpatialCoordinateContract.si().spatial_id,
    )


def _assert_refined_with_boundary_splits(construction: VolumeConstruction) -> None:
    assert construction.boundary_policy == "conforming"
    assert construction.refinement_status is MeshcoreStatus.OK
    assert construction.improvement_status is MeshcoreStatus.OK
    assert construction.refinement_unmet == ()
    assert construction.improvement_unmet == ()
    strata = np.asarray(construction.witness_strata)
    # Facet and segment points were inserted by boundary splits, each on an
    # exact source row.
    assert np.count_nonzero(strata == 1) > 0
    assert np.count_nonzero(strata == 2) > 0


_CUBE_SEED = (
    RegionSeed(
        np.zeros(3, dtype=np.float64), "solid", "solid-material", RegionRole.SOLID
    ),
)


def test_conforming_refinement_splits_grouped_cube_faces_across_source_triangles() -> (
    None
):
    # Facet splits land in whichever source triangle of a face group holds
    # them, and a size-split midpoint that encroaches a subsegment splits the
    # subsegment first, so the size field is met with every cell within bound.
    source = _grouped_cube_source("conforming", 0.0)
    construction = _conforming_refinement(source, _CUBE_SEED, 0.05)
    _assert_refined_with_boundary_splits(construction)
    rows = np.asarray(construction.witness_entities)[
        np.asarray(construction.witness_strata) == 2
    ]
    # Both source triangles of the face groups receive facet points.
    assert np.unique(rows).size == construction.source_faces.shape[0]
    complex_ = source.complex
    diagonal_ids = {
        complex_.plc_edge_ids[row]
        for row, facets in enumerate(complex_.plc_edge_facets)
        if len(facets) != 2
    }
    assert not set(construction.segment_source_ids) & diagonal_ids


def test_conforming_refinement_keeps_undeclared_rotated_cube_triangles_as_authored() -> (
    None
):
    # Without a declared exact plane, the rotated cube's two rounded triangles
    # per face are not coplanar: they are the geometry, and their diagonals
    # stay segments.
    source = _grouped_cube_source("conforming", 0.3)
    construction = _conforming_refinement(source, _CUBE_SEED, 0.05)
    _assert_refined_with_boundary_splits(construction)
    complex_ = source.complex
    diagonal_ids = {
        complex_.plc_edge_ids[row]
        for row, facets in enumerate(complex_.plc_edge_facets)
        if len(facets) != 2
    }
    assert len(diagonal_ids) == 6
    assert diagonal_ids <= set(construction.segment_source_ids)


def _rotated_bipyramid(
    declared: bool,
) -> tuple[NativePlcSource, tuple[RegionSeed, ...]]:
    angle = 0.3
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    points = _POINTS @ rotation.T / 3.0
    # Each triangular facet declares the plane through its rounded corners
    # (first corner, edge vectors rounded) with a tolerance covering them.
    planes = np.asarray(
        [
            (points[a], points[b] - points[a], points[c] - points[a])
            for a, b, c in _POLYGONS.tolist()
        ]
    )
    base = _source("conforming")
    complex_ = PiecewiseLinearComplex(
        points,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=base.complex.vertex_ids,
        vertex_occurrence_paths=base.complex.vertex_occurrence_paths,
        facet_ids=base.complex.facet_ids,
        facet_occurrence_paths=base.complex.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
        boundary="conforming",
        facet_planes=planes if declared else None,
        facet_plane_tolerances=np.full((_POLYGONS.shape[0],), 1e-12)
        if declared
        else None,
    )
    source = NativePlcSource(complex_, "rotated-bipyramid", "revision-1")
    seeds = tuple(
        RegionSeed(
            np.asarray(seed.point) @ rotation.T / 3.0,
            seed.region_name,
            seed.material_id,
            seed.role,
        )
        for seed in _request(source).region_seeds
    )
    return source, seeds


def test_conforming_refinement_meets_bound_in_rotated_bipyramid() -> None:
    # Acute input dihedral angles (54.7 degrees) and segment angles (45
    # degrees) next to boundary splits of every facet.
    source, seeds = _rotated_bipyramid(False)
    _assert_refined_with_boundary_splits(_conforming_refinement(source, seeds, 0.1))


def _assert_split_on_declared_planes(construction: VolumeConstruction) -> None:
    assert construction.refinement_status is MeshcoreStatus.OK
    assert construction.improvement_status is MeshcoreStatus.OK
    assert construction.refinement_unmet == ()
    assert construction.improvement_unmet == ()
    strata = np.asarray(construction.witness_strata)
    # Facet points lie on the declared planes (stratum 3), never on a rounded
    # source triangle; segment points on their source edges.
    assert np.count_nonzero(strata == 1) > 0
    assert np.count_nonzero(strata == 2) == 0
    assert np.count_nonzero(strata == 3) > 0


def test_conforming_refinement_splits_declared_rotated_cube_faces_on_their_planes() -> (
    None
):
    # Declared exact planes group each rotated face's two rounded triangles:
    # the internal diagonals are no segments and refinement meets the bound.
    source = _declared_rotated_cube(
        _rotated_cube_planes(), np.full((6,), 1e-12), "conforming"
    )
    complex_ = source.complex
    construction = _conforming_refinement(source, _CUBE_SEED, 0.05)
    _assert_split_on_declared_planes(construction)
    boundary_ids = {
        complex_.plc_edge_ids[row]
        for row, facets in enumerate(complex_.plc_edge_facets)
        if len(facets) == 2
    }
    assert set(construction.segment_source_ids) == boundary_ids
    # Plane points are facet-stratum vertices of their row's facet, with the
    # facet's region sides and their nonzero rounding deviation.
    vertex_association = _source_associations(source, construction)[0]
    on_plane = np.flatnonzero(np.asarray(construction.witness_strata) == 3)
    facets = np.asarray(construction.source_face_facets)[
        np.asarray(construction.witness_entities)[on_plane]
    ]
    assert all(
        vertex_association.source_roles[row] is GeometrySourceEntityRole.FACET
        for row in on_plane.tolist()
    )
    assert [construction.vertex_source_ids[row] for row in on_plane.tolist()] == [
        complex_.facet_ids[facet] for facet in facets.tolist()
    ]
    assert all(
        vertex_association.source_region_sides[row] != (None, None)
        for row in on_plane.tolist()
    )
    assert not vertex_association.exact


def test_conforming_refinement_splits_declared_rotated_bipyramid_on_its_planes() -> None:
    source, seeds = _rotated_bipyramid(True)
    _assert_split_on_declared_planes(_conforming_refinement(source, seeds, 0.1))


def _conforming_route(source: NativePlcSource, size: float) -> CellMeshingResult:
    base = _request(source, region_seeds=_CUBE_SEED)
    specification = VolumeMeshingSpec(
        base.target,
        base.boundary_scope,
        base.fill_strategy,
        size_controls=(UniformSizeControl(base.boundary_scope, size),),
        region_seeds=base.region_seeds,
    )
    options = NativeMeshingOptions(
        "plc_tetrahedral",
        volume_schedule=NativeVolumeSchedule(
            radius_edge_bound=2.0, minimum_dihedral_degrees=0.0
        ),
    )
    return (
        NativeMeshingProvider(options)
        .plan(source, specification, coordinate_contract=SpatialCoordinateContract.si())
        .execute()
    )


def _rotated_cube(declared: bool, boundary: PlcBoundaryPolicy) -> NativePlcSource:
    return _declared_rotated_cube(
        _rotated_cube_planes() if declared else None,
        np.full((6,), 1e-12) if declared else None,
        boundary,
    )


@pytest.mark.parametrize("declared", (False, True), ids=("undeclared", "declared"))
def test_conforming_route_certifies_boundary_splits_on_exact_source_points(
    declared: bool,
) -> None:
    # Boundary splits leave carriers off the rounded source triangles; coverage
    # decides the chain on the exact source points that the carriers round.
    result = _conforming_route(_rotated_cube(declared, "conforming"), 0.3)
    report = result.certification
    assert report is not None
    assert report.status == "certified"
    witness = report.inputs.source_coverage_witness
    assert isinstance(witness, PlcSourcePointWitness)
    kinds = np.asarray(witness.vertex_kinds)
    assert np.count_nonzero(kinds == 1) > 0
    assert bool(np.any(kinds == 3)) == declared
    coverage = report.coverage
    assert coverage is not None
    assert coverage.status == "certified"
    assert coverage.findings == ()
    assert coverage.source_distance_upper is not None
    if declared:
        # A face's rounded triangles leave its declared plane: the chain is
        # proved exactly in the plane chart, a bounded identification.
        assert coverage.coverage_method == "source_boundary_homotopy"
        assert 0.0 < coverage.source_distance_upper < 1e-12
    else:
        assert coverage.coverage_method == "exact_oriented_facets"
        assert coverage.achieved_region_measures == coverage.requested_region_measures
        # Only the carriers' rounding separates the mesh from the source chain.
        assert 0.0 < coverage.source_distance_upper < 1e-15
    restored = type(coverage).from_stored_fields(coverage)
    assert restored.certificate_id == coverage.certificate_id


def test_unsplit_plc_route_retains_no_source_point_witness() -> None:
    result = _conforming_route(_rotated_cube(True, "fixed"), 5.0)
    report = result.certification
    assert report is not None
    assert report.inputs.source_coverage_witness is None
    assert report.coverage is not None
    assert report.coverage.coverage_method == "exact_oriented_facets"
    assert report.coverage.source_distance_upper is None


@pytest.mark.parametrize(
    ("kind", "shift", "check"),
    (
        (1, 1e-9, "source_point_rounding"),
        (1, 2.0, "source_point_parameter"),
        (3, 1e-9, "source_point_rounding"),
    ),
    ids=("segment-moved", "segment-outside", "plane-moved"),
)
def test_tampered_source_point_witness_is_refused(
    kind: int, shift: float, check: str
) -> None:
    result = _conforming_route(_rotated_cube(True, "conforming"), 0.3)
    report = result.certification
    assert report is not None
    inputs = report.inputs
    witness = inputs.source_coverage_witness
    assert isinstance(witness, PlcSourcePointWitness)
    assert inputs.domain is not None
    assert inputs.cell_regions is not None
    row = int(np.flatnonzero(np.asarray(witness.vertex_kinds) == kind)[0])
    parameters = np.array(witness.vertex_parameters, copy=True)
    parameters[row, 0] += shift
    tampered = PlcSourcePointWitness(
        witness.vertex_global_ids,
        witness.vertex_kinds,
        witness.vertex_entities,
        parameters,
        segments=witness.segments,
        faces=witness.faces,
        face_groups=witness.face_groups,
        face_planes=witness.face_planes,
        face_plane_tolerances=witness.face_plane_tolerances,
        domain_face_rows=witness.domain_face_rows,
        domain_id=witness.domain_id,
        vertex_deviations=witness.vertex_deviations,
    )
    coverage = certify_domain_coverage(
        inputs.mesh,
        inputs.geometry,
        inputs.domain,
        inputs.cell_regions,
        embedding=report.embedding,
        source_coverage_witness=tampered,
    )
    assert coverage.status != "certified"
    vertex = int(witness.vertex_global_ids[row])
    assert any(
        finding.check == check and finding.entity_ids == (vertex,)
        for finding in coverage.findings
    )


def test_mapped_carrier_classification_uses_plc_incidence_and_witnesses() -> None:
    from phydrax.meshing._plc_mapped_support import classify_plc_carriers

    source = _source("fixed")
    construction = _construct(source, _request(source))
    native_vertex_ids = np.asarray(construction.mesh.vertex_global_ids, dtype=np.int64)
    original_vertex_row = int(np.flatnonzero(construction.input_vertices >= 0)[0])
    source_face = np.asarray(construction.boundary_faces[0], dtype=np.int64)
    source_cell = np.asarray(construction.mesh.blocks[0].vertices[0], dtype=np.int64)
    carriers = np.full((3, 4), -1, dtype=np.int64)
    carriers[0, 0] = native_vertex_ids[original_vertex_row]
    carriers[1, :3] = native_vertex_ids[source_face]
    carriers[2] = native_vertex_ids[source_cell]

    ancestry = classify_plc_carriers(construction, carriers)
    assert ancestry.dimensions.tolist() == [
        int(construction.vertex_dimension[original_vertex_row]),
        2,
        3,
    ]
    assert (
        ancestry.source_entity_ids[0]
        == construction.vertex_source_ids[original_vertex_row]
    )
    assert ancestry.source_entity_ids[1] == construction.face_source_ids[0]
    region = int(construction.cell_regions[0])
    assert ancestry.source_entity_ids[2] == construction.region_ids[region]
    assert (
        ancestry.source_occurrence_paths[2]
        == construction.region_occurrence_paths[region]
    )


def _stacked_box_source() -> NativePlcSource:
    points = np.asarray(
        [
            (x, y, z)
            for z in (0.0, 0.5, 1.0)
            for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        ],
        dtype=np.float64,
    )

    def sides(base: int, /) -> tuple[tuple[int, int, int], ...]:
        rows: list[tuple[int, int, int]] = []
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
            a, b = base + first, base + second
            rows.extend(((a, b, b + 4), (a, b + 4, a + 4)))
        return tuple(rows)

    triangles = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (8, 9, 10),
            (8, 10, 11),
            (4, 5, 6),
            (4, 6, 7),
            *sides(0),
            *sides(4),
        ),
        dtype=np.int64,
    )
    # Facets: bottom, top, the two-triangle material interface, then the four
    # side squares of the lower and of the upper box.
    facets = np.asarray(
        (0, 0, 1, 1, 2, 2, *np.repeat(np.arange(3, 11), 2).tolist()),
        dtype=np.int64,
    )
    facet_regions = np.asarray(
        ((-1, 1), (-1, 0), (0, 1), *((-1, 1),) * 4, *((-1, 0),) * 4),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        points,
        triangles,
        facets,
        facet_regions,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"stack-vertex-{row}" for row in range(points.shape[0])),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=tuple(f"stack-facet-{row}" for row in range(11)),
        facet_occurrence_paths=((),) * 11,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
        boundary="fixed",
    )
    return NativePlcSource(complex_, "stacked-boxes", "revision-1")


def test_mapped_carrier_classification_assigns_interface_diagonals_to_their_facet() -> (
    None
):
    from phydrax.meshing._plc_mapped_support import classify_plc_carriers

    source = _stacked_box_source()
    construction = _construct(
        source,
        _request(
            source,
            region_seeds=(
                RegionSeed(
                    np.asarray((0.5, 0.5, 0.75), dtype=np.float64),
                    "upper",
                    "upper-material",
                    RegionRole.SOLID,
                ),
                RegionSeed(
                    np.asarray((0.5, 0.5, 0.25), dtype=np.float64),
                    "lower",
                    "lower-material",
                    RegionRole.SOLID,
                ),
            ),
        ),
    )
    interface_rows = np.flatnonzero(
        np.asarray(construction.face_source_facets, dtype=np.int64) == 2
    )
    interface_faces = np.asarray(construction.boundary_faces, dtype=np.int64)[
        interface_rows
    ]
    assert interface_faces.shape == (2, 3)
    diagonal = sorted(set(interface_faces[0].tolist()) & set(interface_faces[1].tolist()))
    assert len(diagonal) == 2
    native_vertex_ids = np.asarray(construction.mesh.vertex_global_ids, dtype=np.int64)
    carriers = np.full((1, 4), -1, dtype=np.int64)
    carriers[0, :2] = native_vertex_ids[np.asarray(diagonal, dtype=np.int64)]

    ancestry = classify_plc_carriers(construction, carriers)

    assert ancestry.dimensions.tolist() == [2]
    assert ancestry.source_entity_ids == (
        construction.face_source_ids[int(interface_rows[0])],
    )
    assert ancestry.source_entity_ids[0] == source.complex.facet_ids[2]

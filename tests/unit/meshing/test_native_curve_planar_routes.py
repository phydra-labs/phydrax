#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry import BoundaryAtlas, BoundaryMap
from phydrax.geometry._curve_boundary import (
    ParametricCurveBoundarySource,
    SegmentChartCurve,
)
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from phydrax.geometry._source_fidelity import certify_source_fidelity
from phydrax.geometry.brep._patches import LineCurve
from phydrax.geometry.simplicial import PlanarMeshRegion, SegmentMesh
from phydrax.meshing._association import (
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    CurveJunction,
    CurveMeshingSpec,
    MeshingFailure,
    SurfaceMeshingSpec,
)
from phydrax.meshing._controls import FeatureKind, ProtectedFeature
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import (
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing.providers._native_curve import prepare_native_curve
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_planar import prepare_native_planar
from phydrax.meshing.providers._native_sources import (
    NativeCurveSource,
    NativePlanarSource,
)
from phydrax.meshing._canonical import _accept_native_construction, _NATIVE_PROVIDER
from phydrax.meshing._motion import MeshMotionMonitor, advance_mesh_motion
from phydrax.meshing._motion_source import NativePlanarMotionSource


class _VariableSpeedLine(BoundaryMap):
    @property
    def num_charts(self) -> int:
        return 1

    @property
    def reference_dimension(self) -> int:
        return 1

    @property
    def ambient_dimension(self) -> int:
        return 2

    def map(self, chart_indices: Any, reference: Any, /) -> Any:
        del chart_indices
        parameter = jnp.asarray(reference)[..., 0]
        return jnp.stack(
            (parameter + 0.5 * parameter**2, jnp.zeros_like(parameter)),
            axis=-1,
        )

    def jacobian(self, chart_indices: Any, reference: Any, /) -> Any:
        del chart_indices
        parameter = jnp.asarray(reference)[..., 0]
        tangent = jnp.stack((1.0 + parameter, jnp.zeros_like(parameter)), axis=-1)
        return tangent[..., :, None]


def test_curve_route_places_a_nonuniform_chart_by_physical_arc_length() -> None:
    atlas = BoundaryAtlas(
        _VariableSpeedLine(),
        source_entity_ids=jnp.asarray((73,), dtype=jnp.int32),
        source_id="variable-speed-line",
    )
    source = NativeCurveSource(atlas, "revision")
    scope_ids = np.asarray(source.chart_scope_entity_ids, dtype=np.int64)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        1,
        source.scope_entity_set_id(1),
        scope_ids,
    )
    specification = CurveMeshingSpec(
        CellMeshingTarget(1, 2, CellFamilyPolicy(required=("interval",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.5, strength=SizeControlStrength.SOFT),
        ),
        size_compliance=SizeCompliancePolicy(
            absolute_tolerance=0.01,
            target_statistics=("p50", "p95"),
        ),
        maximum_deviation=0.001,
    )
    construction = prepare_native_curve(
        source,
        specification,
        NativeMeshingOptions("curve_arc_length"),
        SpatialCoordinateContract.si(),
    ).execute()
    certification = construction.certification
    fidelity_source = certification.fidelity_source
    fidelity_witness = certification.fidelity_witness
    assert isinstance(fidelity_source, ParametricCurveBoundarySource)
    assert isinstance(fidelity_witness, SourceBoundaryChartWitness)
    entity_key = source.source_entity_key_for_scope(1, int(scope_ids[0]))
    fidelity_source.validate_scope(source, scope_ids, (entity_key[1],))
    assert fidelity_witness.source_query_id == fidelity_source.query_id
    coordinates = np.asarray(construction.mesh.coordinates)
    for expected in (0.0, 0.5, 1.0, 1.5):
        assert np.min(np.abs(coordinates[:, 0] - expected)) < 0.002
    association = next(
        value
        for value in construction.organization.associations
        if value.association_kind is GeometryAssociationKind.CURVE
    )
    intervals = association.source_parameter_intervals
    if intervals is None:
        raise AssertionError("Curve association omitted its source intervals.")
    for interval in intervals:
        if interval is None:
            raise AssertionError("Curve association omitted an authored interval.")
        first, last = interval
        assert first < last
    assert construction.compliance.passed


def test_curve_query_rejects_a_narrowed_authored_feature_range() -> None:
    segments = SegmentMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64),
        np.asarray(((0, 1),), dtype=np.int32),
        source_id="narrowed-curve-query",
    )
    source = NativeCurveSource(
        segments,
        "curve-revision",
        curve_entity_ids=np.asarray((31,), dtype=np.int64),
        vertex_entity_ids=np.asarray((41, 42), dtype=np.int64),
    )
    scope_id = int(source.chart_scope_entity_ids[0])
    entity_id, occurrence = source.source_entity_key_for_scope(1, scope_id)
    query = ParametricCurveBoundarySource(
        (LineCurve(np.asarray((0.0, 0.0)), np.asarray((1.0, 0.0))),),
        ((0.25, 0.75),),
        source_id=source.source_id,
        source_revision=source.source_revision,
        curve_entity_ids=(entity_id,),
        occurrence_paths=(occurrence,),
        chart_ids=(source.chart_ids[0],),
    )

    with pytest.raises(ValueError, match="full parameter range"):
        query.validate_scope(
            source, np.asarray((scope_id,), dtype=np.int64), (occurrence,)
        )


def _curve_request() -> tuple[NativeCurveSource, CurveMeshingSpec, NativeMeshingOptions]:
    segments = SegmentMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1), (1, 2)), dtype=np.int32),
        source_id="native-curve-chain",
    )
    source = NativeCurveSource(
        segments,
        "curve-revision",
        curve_entity_ids=np.asarray((101, 202), dtype=np.int64),
        vertex_entity_ids=np.asarray((501, 602, 703), dtype=np.int64),
    )
    scope_ids = np.asarray(source.chart_scope_entity_ids, dtype=np.int64)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        1,
        source.scope_entity_set_id(1),
        scope_ids,
    )
    target = CellMeshingTarget(1, 2, CellFamilyPolicy(required=("interval",)))
    size = UniformSizeControl(scope, 0.25, strength=SizeControlStrength.HARD)
    corner_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        0,
        source.scope_entity_set_id(0),
        np.asarray((602,), dtype=np.int64),
    )
    corner = ProtectedFeature(
        corner_scope,
        FeatureKind.CORNER,
        maximum_deviation=0.05,
        hard=True,
    )
    curve_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        1,
        source.scope_entity_set_id(1),
        np.asarray((int(scope_ids[0]),), dtype=np.int64),
    )
    curve_feature = ProtectedFeature(
        curve_scope,
        FeatureKind.CURVE,
        maximum_deviation=0.05,
        hard=True,
    )

    specification = CurveMeshingSpec(
        target,
        scope,
        protected_features=(corner, curve_feature),
        size_controls=(size,),
        junctions=(
            CurveJunction(
                "shared-vertex",
                ((int(scope_ids[0]), "end"), (int(scope_ids[1]), "start")),
            ),
        ),
        size_compliance=SizeCompliancePolicy(
            absolute_tolerance=0.01,
            target_statistics=("p50", "p95"),
        ),
        maximum_deviation=0.05,
    )
    return (
        source,
        specification,
        NativeMeshingOptions("curve_arc_length"),
    )


def test_curve_route_places_by_density_and_preserves_source_junction() -> None:
    source, specification, options = _curve_request()
    construction = prepare_native_curve(
        source,
        specification,
        options,
        SpatialCoordinateContract.si(),
    ).execute()
    association = next(
        value
        for value in construction.organization.associations
        if value.association_kind is GeometryAssociationKind.CURVE
    )
    expected = tuple(
        source.source_entity_key_for_scope(1, int(scope_id))[0]
        for scope_id in source.chart_scope_entity_ids
    )
    assert set(association.source_entity_ids) == set(expected)
    intervals = association.source_parameter_intervals
    if intervals is None:
        raise AssertionError("Curve association omitted its source intervals.")
    for interval in intervals:
        if interval is None:
            raise AssertionError("Curve association omitted an authored interval.")
        first, last = interval
        assert isinstance(first, Fraction) and isinstance(last, Fraction)
        assert first < last
    mesh = construction.mesh
    vertex_associations = tuple(
        value
        for value in construction.organization.associations
        if value.association_kind is GeometryAssociationKind.CURVE
        and value.target_entity_set_id == mesh.entity_set(0).entity_set_id
    )
    assert len(vertex_associations) == 2
    for value in vertex_associations:
        points = value.source_parameter_points
        assert points is not None
        assert all(
            isinstance(parameter, Fraction) and Fraction(0) <= parameter <= Fraction(1)
            for parameter in points
        )
    assert {
        source_id
        for value in vertex_associations
        for source_id in value.source_entity_ids
    } == set(expected)
    source_vertex_associations = tuple(
        value
        for value in construction.organization.associations
        if value.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
        and value.target_entity_set_id == mesh.entity_set(0).entity_set_id
    )
    expected_source_vertices = {
        source.source_entity_key_for_scope(0, value)[0] for value in (501, 602, 703)
    }
    assert {
        source_id
        for value in source_vertex_associations
        for source_id in value.source_entity_ids
    } == expected_source_vertices
    block = mesh.blocks[0]
    connectivity = np.asarray(block.vertices, dtype=np.int64)
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    curve_cells = {
        source_id: connectivity[
            np.isin(
                cell_ids,
                np.asarray(association.target_global_ids)[
                    np.asarray(association.source_entity_ids) == source_id
                ],
            )
        ]
        for source_id in expected
    }
    first_vertices = set(curve_cells[expected[0]].reshape(-1).tolist())
    second_vertices = set(curve_cells[expected[1]].reshape(-1).tolist())
    assert len(first_vertices & second_vertices) == 1
    assert any(
        label.name == "shared-vertex" for label in construction.organization.labels
    )
    assert construction.compliance.passed
    assert any(
        label.name == f"feature:{specification.protected_features[0].feature_id}"
        for label in construction.organization.labels
    )
    scoped = construction.certification.scoped_fidelity
    assert len(scoped) == 2
    point_query, point_witness, _ = next(
        row for row in scoped if isinstance(row[1], SourceBoundaryPointWitness)
    )
    chart_query, chart_witness, _ = next(
        row for row in scoped if isinstance(row[1], SourceBoundaryChartWitness)
    )
    assert isinstance(point_query, ParametricCurveBoundarySource)
    assert isinstance(point_witness, SourceBoundaryPointWitness)
    assert isinstance(chart_query, ParametricCurveBoundarySource)
    assert isinstance(chart_witness, SourceBoundaryChartWitness)
    assert point_witness.source_query_id == point_query.query_id
    point_key = source.source_entity_key_for_scope(0, 602)
    lower, upper = point_query.source_point_bounds(*point_key)
    np.testing.assert_array_equal(lower, np.asarray((1.0, 0.0)))
    np.testing.assert_array_equal(upper, np.asarray((1.0, 0.0)))
    source.validate_point_boundary_query(point_query, np.asarray((602,), dtype=np.int64))
    curve_id = int(source.chart_scope_entity_ids[0])
    curve_key = source.source_entity_key_for_scope(1, curve_id)
    chart_query.validate_scope(
        source, np.asarray((curve_id,), dtype=np.int64), (curve_key[1],)
    )
    assert chart_witness.source_query_id == chart_query.query_id
    assert set(chart_witness.source_entity_ids) == {
        source.source_entity_key_for_scope(1, int(source.chart_scope_entity_ids[0]))[0]
    }


def test_segment_curve_query_is_an_exact_affine_chart_on_its_declared_range() -> None:
    segments = SegmentMesh(
        np.asarray(((0.1, 0.0), (1.1, 0.3)), dtype=np.float64),
        np.asarray(((0, 1),), dtype=np.int32),
        source_id="exact-segment-chart",
    )
    source = NativeCurveSource(
        segments,
        "revision-1",
        curve_entity_ids=np.asarray((1,), dtype=np.int64),
        vertex_entity_ids=np.asarray((1, 2), dtype=np.int64),
        chart_parameter_ranges=np.asarray(((0.3, 1.7),), dtype=np.float64),
    )
    scope = MeshingScope.whole(source, 1)
    specification = CurveMeshingSpec(
        CellMeshingTarget(1, 2, CellFamilyPolicy(required=("interval",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.2, strength=SizeControlStrength.SOFT),
        ),
        maximum_deviation=1.0e-8,
    )
    construction = prepare_native_curve(
        source,
        specification,
        NativeMeshingOptions("curve_arc_length"),
        SpatialCoordinateContract.si(),
    ).execute()
    assert construction.compliance.passed
    query = construction.certification.fidelity_source
    witness = construction.certification.fidelity_witness
    assert isinstance(query, ParametricCurveBoundarySource)
    assert isinstance(witness, SourceBoundaryChartWitness)
    (carrier,) = query.curves
    assert isinstance(carrier, SegmentChartCurve)
    start = tuple(Fraction(value) for value in (0.1, 0.0))
    end = tuple(Fraction(value) for value in (1.1, 0.3))
    assert carrier.evaluate_exact(Fraction(0.3)) == start
    assert carrier.evaluate_exact(Fraction(1.7)) == end
    np.testing.assert_array_equal(
        np.asarray(carrier.evaluate(jnp.asarray((0.3, 1.7)))),
        np.asarray(((0.1, 0.0), (1.1, 0.3))),
    )

    certificate = certify_source_fidelity(
        construction.mesh,
        construction.geometry,
        query,
        tolerance=1.0e-8,
        witness=witness,
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper <= 1.0e-8
    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    length = float(np.hypot(1.0, 0.3))
    cells = math.ceil(length / 0.2)
    expected = np.asarray(start, dtype=np.float64) + np.linspace(0.0, 1.0, cells + 1)[
        :, None
    ] * np.asarray((1.0, 0.3))
    order = np.argsort(coordinates[:, 0])
    np.testing.assert_allclose(coordinates[order], expected, rtol=0.0, atol=1.0e-7)


def _planar_request() -> tuple[
    NativePlanarSource, SurfaceMeshingSpec, NativeMeshingOptions
]:
    region = PlanarMeshRegion(
        np.asarray(
            (
                (0.0, 0.0),
                (2.0, 0.0),
                (2.0, 2.0),
                (0.0, 2.0),
                (0.75, 0.75),
                (0.75, 1.25),
                (1.25, 1.25),
                (1.25, 0.75),
            ),
            dtype=np.float64,
        ),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        feature_id="native-planar-hole",
    )
    embedded = SegmentMesh(
        np.asarray(((0.25, 1.5), (0.5, 1.5)), dtype=np.float64),
        np.asarray(((0, 1),), dtype=np.int32),
        source_id="native-planar-feature",
    )
    region_edge_ids = np.asarray((110, 120, 130, 140, 150, 160, 170, 180), dtype=np.int64)
    embedded_edge_id = 910
    source = NativePlanarSource(
        region,
        "planar-revision",
        embedded=embedded,
        region_scope_entity_ids=np.asarray((407,), dtype=np.int64),
        vertex_entity_ids=np.asarray((11, 12, 13, 14, 15, 16, 17, 18), dtype=np.int64),
        edge_entity_ids=region_edge_ids,
        embedded_vertex_entity_ids=np.asarray((71, 72), dtype=np.int64),
        embedded_edge_entity_ids=np.asarray((embedded_edge_id,), dtype=np.int64),
    )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    target = CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",)))
    control = UniformSizeControl(scope, 0.5, strength=SizeControlStrength.SOFT)
    edge_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        1,
        source.scope_entity_set_id(1),
        np.asarray((embedded_edge_id,), dtype=np.int64),
    )
    feature = ProtectedFeature(
        edge_scope,
        FeatureKind.CURVE,
        maximum_deviation=0.05,
        hard=True,
    )
    corner_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        0,
        source.scope_entity_set_id(0),
        np.asarray((71,), dtype=np.int64),
    )
    corner_feature = ProtectedFeature(
        corner_scope,
        FeatureKind.CORNER,
        maximum_deviation=0.05,
        hard=True,
    )
    specification = SurfaceMeshingSpec(
        target,
        scope,
        planar_embedding=None,
        size_controls=(control,),
        protected_features=(feature, corner_feature),
        size_compliance=SizeCompliancePolicy(
            absolute_tolerance=0.05,
            target_statistics=("p50",),
        ),
    )
    return source, specification, NativeMeshingOptions("planar_constrained_delaunay")


def test_planar_motion_requires_authored_successor_and_preserves_scopes() -> None:
    authored, specification, options = _planar_request()
    construction = prepare_native_planar(
        authored, specification, options, SpatialCoordinateContract.si()
    ).execute()
    source = _accept_native_construction(
        construction, specification, _NATIVE_PROVIDER, "test", "test", "test"
    )
    monitor = MeshMotionMonitor(source.mesh)
    shift = np.asarray((0.55, 0.0))
    coordinates = np.asarray(source.mesh.coordinates) + shift
    transition = NativePlanarMotionSource(authored, shift)
    accepted = advance_mesh_motion(
        monitor, source, coordinates, boundary_residual=0.0, source_transition=transition
    )
    assert accepted.accepted, tuple(
        (a.reasons, a.failure_detail) for a in accepted.assessments
    )
    assert accepted.result is not None and accepted.result.certification is not None
    assert accepted.source_transition is transition
    inputs = accepted.result.certification.inputs
    assert isinstance(inputs.feature_source, NativePlanarSource)
    assert inputs.feature_source.binding_id == transition.successor.binding_id
    assert inputs.source_revision == transition.successor.source_revision
    assert (
        NativePlanarMotionSource.from_stored_fields(transition).transition_id
        == transition.transition_id
    )
    assert all(
        a.source_revision == transition.successor.source_revision
        for a in accepted.result.associations
    )
    np.testing.assert_array_equal(
        np.asarray(transition.successor.region.vertices),
        np.asarray(authored.region.vertices) + shift,
    )
    assert source.certification is not None
    assert inputs.protected_features is not None
    assert source.certification.inputs.protected_features is not None
    assert len(inputs.scoped_fidelity) == len(source.certification.inputs.scoped_fidelity)
    assert {tuple(f.scope.entity_ids.tolist()) for f in inputs.protected_features} == {
        tuple(f.scope.entity_ids.tolist())
        for f in source.certification.inputs.protected_features
    }
    second_shift = np.asarray((0.05, 0.0))
    second_authority = NativePlanarMotionSource(transition.successor, second_shift)
    second = advance_mesh_motion(
        MeshMotionMonitor(accepted.result.mesh),
        accepted.result,
        np.asarray(accepted.result.mesh.coordinates) + second_shift,
        boundary_residual=0.0,
        source_transition=second_authority,
    )
    assert second.accepted, tuple(
        (a.reasons, a.failure_detail) for a in second.assessments
    )
    assert second.result is not None and second.result.certification is not None
    assert (
        second.result.certification.inputs.source_revision
        == second_authority.successor.source_revision
    )
    fixed = advance_mesh_motion(monitor, source, coordinates, boundary_residual=0.0)
    assert not fixed.accepted and fixed.result is None
    wrong = NativePlanarMotionSource(transition.successor, shift)
    stale = advance_mesh_motion(
        monitor, source, coordinates, boundary_residual=0.0, source_transition=wrong
    )
    assert not stale.accepted and stale.result is None
    changed_scope = NativePlanarSource(
        authored.region,
        authored.declared_source_revision,
        embedded=authored.embedded,
        region_scope_entity_ids=np.asarray((408,), dtype=np.int64),
        vertex_entity_ids=authored.vertex_entity_ids,
        edge_entity_ids=authored.edge_entity_ids,
        embedded_vertex_entity_ids=authored.embedded_vertex_entity_ids,
        embedded_edge_entity_ids=authored.embedded_edge_entity_ids,
    )
    wrong_scope = advance_mesh_motion(
        monitor,
        source,
        coordinates,
        boundary_residual=0.0,
        source_transition=NativePlanarMotionSource(changed_scope, shift),
    )
    assert not wrong_scope.accepted and wrong_scope.result is None
    noncommuting = coordinates.copy()
    noncommuting[-1, 0] += 0.01
    refused = advance_mesh_motion(
        monitor, source, noncommuting, boundary_residual=0.0, source_transition=transition
    )
    assert not refused.accepted and refused.result is None


def test_planar_motion_composition_cold_replay_and_tampering(tmp_path: Path) -> None:
    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax._model._structure import model_archive_projection
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        MODEL_ARCHIVE_LIMITS,
        model_archive_content_id,
        planar_motion_advance_recipe,
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )

    authored, specification, options = _planar_request()
    construction = prepare_native_planar(
        authored, specification, options, SpatialCoordinateContract.si()
    ).execute()
    source = _accept_native_construction(
        construction, specification, _NATIVE_PROVIDER, "test", "test", "test"
    )
    monitor = MeshMotionMonitor(source.mesh)
    translation = np.asarray((0.55, 0.0))
    coordinates = np.asarray(source.mesh.coordinates) + translation
    fixed = np.zeros((coordinates.shape[0],), dtype=np.bool_)
    transition = NativePlanarMotionSource(authored, translation)
    advance = advance_mesh_motion(
        monitor,
        source,
        coordinates,
        boundary_residual=0.0,
        fixed_vertices=fixed,
        numeric_version="planar-motion-archive-test",
        source_transition=transition,
    )
    assert advance.accepted and advance.result is not None
    recipe = planar_motion_advance_recipe(
        monitor,
        source,
        coordinates,
        advance,
        boundary_residual=0.0,
        fixed_vertices=fixed,
        numeric_version="planar-motion-archive-test",
        adaptation_policy=None,
        remesh_metric=None,
        prior_transition=None,
        source_transition=transition,
    )
    predecessor_entry = CompositionEntry(
        source,
        entry_id="motion/predecessor",
        role="topology",
        owner_id=authored.source_id,
        structure_id=source.mesh.topology_id,
        revision_id=source.result_id,
        semantics_id=authored.binding_id,
    )
    target_entry = CompositionEntry(
        advance.result,
        entry_id="motion/target",
        role="topology",
        owner_id=authored.source_id,
        structure_id=advance.result.mesh.topology_id,
        revision_id=advance.result.result_id,
        semantics_id=transition.successor.binding_id,
        dependencies=(predecessor_entry.binding("revision"),),
    )
    entry = CompositionEntry(
        recipe,
        entry_id="motion/advance",
        role="topology",
        owner_id=authored.source_id,
        structure_id=source.mesh.topology_id,
        revision_id=advance.advance_id,
        semantics_id=transition.transition_id,
        dependencies=(
            predecessor_entry.binding("revision"),
            target_entry.binding("revision"),
        ),
    )
    composition = Composition(
        (predecessor_entry, target_entry, entry), boundary_id="planar-motion-cold"
    )
    path = write_model_artifact(
        tmp_path / "motion.phx", composition, limits=MODEL_ARCHIVE_LIMITS
    )
    content_id = model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS)
    script = (
        "import sys; "
        "from phydrax.lifecycle import Composition; "
        "from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS,read_model_artifact,restore_model_archive_value,restore_planar_motion_advance; "
        "archive=read_model_artifact(sys.argv[1],expected_type=Composition,restore=restore_model_archive_value,limits=MODEL_ARCHIVE_LIMITS,expected_content_id=sys.argv[2]); "
        "source=archive.value('motion/predecessor'); target=archive.value('motion/target'); "
        "assert source.result_id==sys.argv[3] and target.result_id==sys.argv[4]; "
        "assert source.certification.passed and target.certification.passed; "
        "recipe=archive.value('motion/advance'); "
        "assert target.certification.inputs.feature_source.binding_id==archive.entry('motion/target').semantics_id; "
        "a=restore_planar_motion_advance(recipe,source); "
        "assert a.accepted and a.advance_id==sys.argv[5] and a.result.result_id==target.result_id and a.source_transition.successor.binding_id==archive.entry('motion/target').semantics_id"
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            content_id,
            source.result_id,
            advance.result.result_id,
            advance.advance_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    restored = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
        expected_content_id=content_id,
    )
    assert restored.value("motion/advance").advance_id == advance.advance_id
    assert restored.value("motion/predecessor").result_id == source.result_id
    assert restored.value("motion/target").result_id == advance.result.result_id
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
            expected_content_id="0" * 64,
        )

    projected = model_archive_projection(transition)
    with pytest.raises(ValueError):
        restore_model_archive_value(
            projected._replace(translation=np.asarray((0.56, 0.0)))
        )
    altered_vertices = np.asarray(projected.source.vertices).copy()
    altered_vertices[0, 0] += 0.01
    with pytest.raises(ValueError):
        restore_model_archive_value(
            projected._replace(
                source=projected.source._replace(vertices=altered_vertices)
            )
        )
    assert source.certification is not None
    query = source.certification.inputs.scoped_fidelity[0][0]
    query_recipe = model_archive_projection(query)
    bounds = list(query_recipe.endpoint_point_bounds)
    first = list(bounds[0])
    first[0] = ((100.0, 100.0), (100.0, 100.0))
    bounds[0] = tuple(first)
    with pytest.raises(ValueError):
        restore_model_archive_value(
            query_recipe._replace(endpoint_point_bounds=tuple(bounds))
        )


def test_planar_cdt_preserves_holes_embedded_constraint_and_exact_ancestry() -> None:
    source, specification, options = _planar_request()
    construction = prepare_native_planar(
        source,
        specification,
        options,
        SpatialCoordinateContract.si(),
    ).execute()
    domain = construction.certification.domain
    assert domain is not None
    assert isinstance(domain, PiecewiseLinearDomain)
    scoped = construction.certification.scoped_fidelity
    assert len(scoped) == 2
    curve_query, curve_witness, curve_tolerance = next(
        row for row in scoped if isinstance(row[1], SourceBoundaryChartWitness)
    )
    point_query, point_witness, point_tolerance = next(
        row for row in scoped if isinstance(row[1], SourceBoundaryPointWitness)
    )
    assert isinstance(curve_query, ParametricCurveBoundarySource)
    assert isinstance(curve_witness, SourceBoundaryChartWitness)
    assert isinstance(point_query, ParametricCurveBoundarySource)
    assert isinstance(point_witness, SourceBoundaryPointWitness)
    assert curve_tolerance == point_tolerance == 0.05
    source.validate_curve_boundary_query(
        curve_query,
        np.asarray((int(source.embedded_edge_entity_ids[0]),), dtype=np.int64),
    )
    curve_query.validate_scope(
        source,
        np.asarray((int(source.embedded_edge_entity_ids[0]),), dtype=np.int64),
        (
            source.source_entity_key_for_scope(
                1, int(source.embedded_edge_entity_ids[0])
            )[1],
        ),
    )
    source.validate_point_boundary_query(point_query, np.asarray((71,), dtype=np.int64))
    point_key = source.source_entity_key_for_scope(0, 71)
    point_lower, point_upper = point_query.source_point_bounds(*point_key)
    np.testing.assert_array_equal(point_lower, np.asarray((0.25, 1.5)))
    np.testing.assert_array_equal(point_upper, np.asarray((0.25, 1.5)))
    assert curve_witness.source_query_id == curve_query.query_id
    assert point_witness.source_query_id == point_query.query_id
    assert np.asarray(domain.facets).shape[0] == source.region.edges.shape[0]
    cell_regions = np.asarray(construction.certification.cell_regions)
    assert cell_regions.shape == (construction.mesh.blocks[0].cell_count,)
    assert np.all(cell_regions == 0)
    coordinates = np.asarray(construction.mesh.coordinates)
    triangles = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
    centroids = np.mean(coordinates[triangles], axis=1)
    assert not np.any(
        (centroids[:, 0] > 0.75)
        & (centroids[:, 0] < 1.25)
        & (centroids[:, 1] > 0.75)
        & (centroids[:, 1] < 1.25)
    )
    names = {label.name for label in construction.organization.labels}
    assert "region:0" in names
    assert "embedded" in names
    assert {patch.name for patch in construction.organization.patches} == {
        "loop:0",
        "loop:1",
    }
    curve_associations = tuple(
        value
        for value in construction.organization.associations
        if value.association_kind is GeometryAssociationKind.CURVE
    )
    intervals_by_source: dict[str, list[tuple[Fraction, Fraction]]] = {}
    edge_set_id = construction.mesh.entity_set(1).entity_set_id
    for association in curve_associations:
        if association.target_entity_set_id != edge_set_id:
            continue
        intervals = association.source_parameter_intervals
        if intervals is None:
            raise AssertionError("Planar curve-edge association omitted exact intervals.")
        for source_id, interval in zip(
            association.source_entity_ids, intervals, strict=True
        ):
            if interval is None:
                raise AssertionError("Resolved planar curve edge omitted its interval.")
            intervals_by_source.setdefault(source_id, []).append(interval)
    expected_edge_ids = tuple(
        source.source_entity_key_for_scope(1, int(edge_id))[0]
        for edge_id in source.scope_entity_ids(1)
    )
    assert set(intervals_by_source) == set(expected_edge_ids)
    for source_id in expected_edge_ids:
        spans = sorted(intervals_by_source[source_id])
        assert spans[0][0] == Fraction(0)
        assert spans[-1][1] == Fraction(1)
        assert all(first[1] == second[0] for first, second in zip(spans[:-1], spans[1:]))
    region_association = next(
        value
        for value in construction.organization.associations
        if value.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
        and value.target_entity_set_id == construction.mesh.entity_set(2).entity_set_id
    )
    assert set(region_association.source_entity_ids) == {source.region_entity_id}
    assert construction.compliance.passed
    vertex_associations = tuple(
        value
        for value in construction.organization.associations
        if value.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
        and value.target_entity_set_id == construction.mesh.entity_set(0).entity_set_id
    )
    observed_vertex_sources = {
        source_id
        for association in vertex_associations
        for source_id, role in zip(
            association.source_entity_ids,
            association.source_roles,
            strict=True,
        )
        if role is GeometrySourceEntityRole.VERTEX
    }
    expected_vertex_sources = {
        source.source_entity_key_for_scope(0, int(vertex_id))[0]
        for vertex_id in source.scope_entity_ids(0)
    }
    assert observed_vertex_sources == expected_vertex_sources
    point_associations = tuple(
        association
        for association in curve_associations
        if association.source_parameter_points is not None
        and any(point is not None for point in association.source_parameter_points)
    )
    assert point_associations
    for association in point_associations:
        points = association.source_parameter_points
        assert points is not None
        assert all(point is None or isinstance(point, Fraction) for point in points)


def test_curve_route_rejects_conflicting_hard_size_controls() -> None:
    source, specification, options = _curve_request()
    conflict = CurveMeshingSpec(
        specification.target,
        specification.scope,
        size_controls=(
            UniformSizeControl(
                specification.scope, 0.25, strength=SizeControlStrength.HARD
            ),
            UniformSizeControl(
                specification.scope, 0.5, strength=SizeControlStrength.HARD
            ),
        ),
        protected_features=specification.protected_features,
        junctions=specification.junctions,
        size_compliance=specification.size_compliance,
        maximum_deviation=specification.maximum_deviation,
    )
    with pytest.raises(MeshingFailure, match="size controls do not resolve"):
        prepare_native_curve(
            source,
            conflict,
            options,
            SpatialCoordinateContract.si(),
        ).execute()


def test_planar_route_rejects_conflicting_hard_size_controls() -> None:
    source, specification, options = _planar_request()
    conflict = SurfaceMeshingSpec(
        specification.target,
        specification.scope,
        planar_embedding=specification.planar_embedding,
        size_controls=(
            UniformSizeControl(
                specification.scope, 0.5, strength=SizeControlStrength.HARD
            ),
            UniformSizeControl(
                specification.scope, 0.25, strength=SizeControlStrength.HARD
            ),
        ),
        protected_features=specification.protected_features,
        size_compliance=specification.size_compliance,
    )
    with pytest.raises(
        MeshingFailure, match="planar region size controls do not resolve"
    ):
        prepare_native_planar(
            source,
            conflict,
            options,
            SpatialCoordinateContract.si(),
        ).execute()


@pytest.mark.parametrize("corruption", ("parameter", "occurrence", "missing-proof"))
def test_curve_junction_audit_requires_each_source_closure(corruption: str) -> None:
    from phydrax.geometry._source_fidelity import validate_source_fidelity_witness
    from phydrax.meshing._association import GeometryAssociation
    from phydrax.meshing._audit import audit_cell_mesh

    source, specification, options = _curve_request()
    construction = prepare_native_curve(
        source, specification, options, SpatialCoordinateContract.si()
    ).execute()
    mesh, geometry = construction.mesh, construction.geometry
    request = construction.certification
    if request.fidelity_source is None or request.fidelity_witness is None:
        raise AssertionError("The source curve fixture requires a fidelity witness.")
    token = validate_source_fidelity_witness(
        mesh, geometry, request.fidelity_source, request.fidelity_witness
    )
    associations = construction.organization.associations
    vertex_curves = tuple(
        value
        for value in associations
        if value.association_kind is GeometryAssociationKind.CURVE
        and value.target_entity_set_id == mesh.entity_set(0).entity_set_id
    )
    first, second = vertex_curves
    shared = set(np.asarray(first.target_global_ids).tolist()) & set(
        np.asarray(second.target_global_ids).tolist()
    )
    assert len(shared) == 1
    row = np.asarray(first.target_global_ids).tolist().index(shared.pop())
    points = list(first.source_parameter_points)
    occurrences = list(first.source_occurrence_paths)
    parameters = np.array(first.parameters, copy=True)
    if corruption == "parameter":
        points[row] = Fraction(1, 2)
        parameters[row, 0] = 0.5
    elif corruption == "occurrence":
        occurrences[row] = ("unrelated-instance",)
    forged = GeometryAssociation(
        first.association_kind,
        first.source_id,
        first.source_revision,
        first.target_entity_set_id,
        first.target_global_ids,
        first.source_entity_ids,
        first.residuals,
        resolved=first.resolved,
        exact=first.exact,
        source_roles=first.source_roles,
        source_occurrence_paths=tuple(occurrences),
        source_parameter_points=tuple(points),
        parameters=parameters,
        orientations=first.orientations,
    )
    report = audit_cell_mesh(
        mesh,
        geometry,
        associations=tuple(forged if value is first else value for value in associations),
        validated_fidelity=() if corruption == "missing-proof" else (token,),
    )
    assert not report.passed
    assert "association_curve_vertex_closure" in report.issues

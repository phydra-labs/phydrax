#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import dataclasses
import math
from fractions import Fraction
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
from phydrax.geometry._approximate_source_coverage import _bound_boundary_source
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._domain_planar_boundary import CadSourceDomain
from phydrax.geometry._embedding_state import (
    coordinate_certificate_request,
    MeshCertificateLimits,
)
from phydrax.geometry._source_fidelity import (
    analytic_boundary_proof,
    certify_source_fidelity,
)
from phydrax.geometry.brep import BRepGeometry, CircleCurve
from phydrax.geometry.brep._analytic_source_regions import complete_cylinder_region
from phydrax.geometry.brep._intersection import NativePeriodEndpoint
from phydrax.meshing._audit import required_audit_policy
from phydrax.meshing.providers._native_cad_volume import (
    NativeCadVolumeSource,
)
from tests._support.cad_curving import CURVING_STATUS, cylinder_mesh
from tests._support.meshing_fixtures import domain_scope


pytestmark = pytest.mark.meshcore

_SI = phx.SpatialCoordinateContract.si()
_RADIUS, _HEIGHT, _SIZE, _DEVIATION = 1.0, 2.0, 0.5, 0.05


def _cylinder() -> phx.geometry.BRepModel:
    return phx.geometry.brep_cylinder(
        _RADIUS,
        _HEIGHT,
        coordinate_contract=_SI,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )


@pytest.fixture(scope="module")
def cylinder_surface() -> tuple[
    phx.geometry.BRepModel,
    phx.meshing.NativeSurfaceSource,
    phx.meshing.CellMeshingResult,
    phx.meshing.MeshingScope,
]:
    """One certified whole-boundary surface shared by every consumer below."""
    model = _cylinder()
    return (model, *_surface(model))


def _surface(
    model: phx.geometry.BRepModel,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> tuple[
    phx.meshing.NativeSurfaceSource, phx.meshing.CellMeshingResult, phx.meshing.MeshingScope
]:
    domain = phx.geometry.MeshingDomain.from_brep(model)
    scope = domain_scope(domain, 2, domain.scope_indices(2))
    source = phx.meshing.NativeSurfaceSource(domain)
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, _SIZE, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        protected_features=(
            phx.meshing.ProtectedFeature(
                scope, phx.meshing.FeatureKind.SURFACE, maximum_deviation=_DEVIATION
            ),
        ),
    )
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("parametric_surface")
        )
        .plan(source, request, coordinate_contract=_SI)
        .execute(coordinate_operands=coordinate_operands)
    )
    return source, result, scope


def _volume_request(
    scope: phx.meshing.MeshingScope, deviation: float = _DEVIATION
) -> phx.meshing.VolumeMeshingSpec:
    return phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, _SIZE, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        protected_features=(
            phx.meshing.ProtectedFeature(
                scope, phx.meshing.FeatureKind.SURFACE, maximum_deviation=deviation
            ),
        ),
    )


def _fill(
    source: NativeCadVolumeSource, request: phx.meshing.VolumeMeshingSpec
) -> phx.meshing.CellMeshingResult:
    return (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("cad_tetrahedral")
        )
        .plan(source, request, coordinate_contract=_SI)
        .execute()
    )


def test_cylinder_fill_publishes_under_root_brep_with_exact_lineage(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
) -> None:
    model, surface_source, surface, scope = cylinder_surface
    source = NativeCadVolumeSource(surface_source, surface)
    assert source.source_id == model.source_id
    assert source.plc_source.source_id != model.source_id
    assert source.plc_source.complex.boundary == "fixed"
    assert surface.certification is not None
    assert source.maximum_deviation == pytest.approx(
        surface.certification.request.fidelity_tolerance
    )
    result = _fill(source, _volume_request(scope))
    receipt = result.execution_evidence
    assert receipt is not None and receipt.preparation_evidence is not None
    assert receipt.preparation_evidence.history_evidence == (surface.execution_evidence,)
    assert receipt.preparation_evidence.preparation_evidence is None
    certificate = result.certification
    assert certificate is not None and certificate.passed
    outcomes = {outcome.check: outcome.status for outcome in certificate.outcomes}
    assert {"source_fidelity", "domain_coverage", "global_embedding"} <= set(outcomes)
    assert certificate.fidelity is not None
    assert certificate.fidelity.mesh_to_source_upper <= _DEVIATION
    assert result.trace.binding is not None
    assert (result.trace.binding.source_id, result.trace.binding.source_revision) == (
        model.source_id,
        model.source_revision,
    )
    assert result.associations
    for association in result.associations:
        assert association.association_kind is phx.meshing.GeometryAssociationKind.BREP
        assert association.source_id == model.source_id
    vertex_set = result.mesh.entity_set(0).entity_set_id
    vertices = next(
        value for value in result.associations if value.target_entity_set_id == vertex_set
    )
    assert np.asarray(vertices.target_global_ids).size == result.mesh.coordinates.shape[0]
    points = np.asarray(result.mesh.coordinates)
    on_boundary = np.asarray(vertices.source_dimensions) < 3
    radial = np.hypot(points[:, 0], points[:, 1])
    lateral = on_boundary & (np.abs(points[:, 2]) > 1e-12) & (
        np.abs(points[:, 2] - _HEIGHT) > 1e-12
    )
    # Fixed-boundary lineage: lateral boundary vertices are on the exact cylinder.
    np.testing.assert_allclose(radial[lateral], _RADIUS, atol=1e-12)
    cells = np.asarray(result.mesh.blocks[0].vertices)
    corners = points[cells]
    volume = float(np.sum(np.linalg.det(corners[:, 1:] - corners[:, :1]))) / 6.0
    assert 0.0 < volume < math.pi * _RADIUS**2 * _HEIGHT
    domain = surface_source.domain
    face_set, cell_set = (result.mesh.entity_set(d).entity_set_id for d in (2, 3))
    faces = next(
        value for value in result.associations if value.target_entity_set_id == face_set
    )
    cells_ = next(
        value for value in result.associations if value.target_entity_set_id == cell_set
    )
    # Every boundary face names a B-Rep face of this occurrence and reproduces
    # one surface triangle, keeping its certified chord residual (never zero on
    # the curved face).
    face_names = faces.source_entities.entity_ids()
    assert set(face_names) == {
        domain.entity_id(2, row) for row in range(len(domain.patches))
    }
    surface_faces = next(
        value
        for value in surface.associations
        if value.target_entity_set_id == surface.mesh.entity_set(2).entity_set_id
    )
    residuals = np.asarray(faces.residuals)
    assert not faces.exact
    np.testing.assert_array_equal(
        np.sort(residuals), np.sort(np.asarray(surface_faces.residuals))
    )
    assert 0.0 < float(np.max(residuals)) <= source.maximum_deviation
    assert set(cells_.source_entities.entity_ids()) == {domain.entity_id(3, 0)}
    assert np.asarray(cells_.target_global_ids).size == cells.shape[0]


def test_sphere_fill_certifies_its_boundary_through_the_root_chart_chain() -> None:
    """No closed-form cylinder theorem applies: the surface's chart chain proves it."""
    radius = 1.0
    model = phx.geometry.brep_sphere(
        radius,
        coordinate_contract=_SI,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    surface_source, surface, scope = _surface(model)
    source = NativeCadVolumeSource(surface_source, surface)
    assert source.fidelity_source.chart_triangulations
    result = _fill(source, _volume_request(scope))
    certificate = result.certification
    assert certificate is not None and certificate.passed
    fidelity = certificate.fidelity
    assert fidelity is not None and fidelity.chart_coverage is not None
    assert (fidelity.mesh_to_source_semantics, fidelity.source_to_mesh_semantics) == (
        "certified",
        "certified",
    )
    assert fidelity.mesh_to_source_upper <= _DEVIATION
    assert fidelity.source_to_mesh_upper <= _DEVIATION
    # Independent oracle: every boundary vertex is on the exact sphere, and the
    # inscribed tetrahedra never exceed the ball.
    points = np.asarray(result.mesh.coordinates)
    vertex_set = result.mesh.entity_set(0).entity_set_id
    vertices = next(
        value for value in result.associations if value.target_entity_set_id == vertex_set
    )
    on_boundary = np.asarray(vertices.source_dimensions) < 3
    np.testing.assert_allclose(
        np.linalg.norm(points[on_boundary], axis=1), radius, atol=1e-12
    )
    corners = points[np.asarray(result.mesh.blocks[0].vertices)]
    volume = float(np.sum(np.linalg.det(corners[:, 1:] - corners[:, :1]))) / 6.0
    assert 0.0 < volume < 4.0 / 3.0 * math.pi * radius**3


def test_tighter_requested_deviation_than_the_surface_refuses_publication(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
) -> None:
    _, surface_source, surface, scope = cylinder_surface
    source = NativeCadVolumeSource(surface_source, surface)
    with pytest.raises(phx.meshing.MeshingFailure, match="source_fidelity"):
        _fill(source, _volume_request(scope, deviation=1e-9))


def test_surface_from_another_revision_is_refused(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
) -> None:
    _, _, surface, _ = cylinder_surface
    other = phx.geometry.brep_cylinder(
        _RADIUS,
        2.0 * _HEIGHT,
        coordinate_contract=_SI,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    with pytest.raises(ValueError, match="bind the B-Rep domain revision"):
        NativeCadVolumeSource(
            phx.meshing.NativeSurfaceSource(phx.geometry.MeshingDomain.from_brep(other)),
            surface,
        )


def test_uncertified_surface_is_refused(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
) -> None:
    _, surface_source, surface, _ = cylinder_surface
    stripped = eqx.tree_at(
        lambda value: value.certification, surface, None, is_leaf=lambda x: x is None
    )
    with pytest.raises(ValueError, match="certified native surface"):
        NativeCadVolumeSource(surface_source, stripped)


def test_stored_source_rederives_its_banks_and_refuses_foreign_identity(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
) -> None:
    _, surface_source, surface, _ = cylinder_surface
    source = NativeCadVolumeSource(surface_source, surface)
    # Stored lineage banks are never trusted: restoration rederives them.
    tampered = eqx.tree_at(
        lambda value: value.face_residuals, source, np.zeros_like(source.face_residuals)
    )
    restored = NativeCadVolumeSource.from_stored_fields(tampered)
    assert restored.binding_id == source.binding_id
    np.testing.assert_array_equal(restored.face_residuals, source.face_residuals)
    assert restored.plc_source.binding_id == source.plc_source.binding_id
    other = phx.geometry.MeshingDomain.from_brep(
        phx.geometry.brep_cylinder(
            _RADIUS,
            2.0 * _HEIGHT,
            coordinate_contract=_SI,
            tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
        )
    )
    foreign = eqx.tree_at(lambda value: value.domain, source, other)
    with pytest.raises(ValueError, match="bind the B-Rep domain revision"):
        NativeCadVolumeSource.from_stored_fields(foreign)


def test_actual_inspected_plan_seal_binds_owners_banks_and_weak_lifetime(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
) -> None:
    import gc
    import weakref
    from copy import copy

    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.meshing.providers import _native as owner

    _, surface_source, surface, scope = cylinder_surface
    source = NativeCadVolumeSource(surface_source, surface)
    request = _volume_request(scope)
    plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("cad_tetrahedral")
    ).plan(source, request, coordinate_contract=_SI)
    assert plan.source is source
    assert plan.specification is request
    with NativeExecutionScope(request.limits.native_limits()) as execution:
        with execution.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=request.limits.maximum_work_units,
                maximum_memory_bytes=request.limits.maximum_scratch_bytes,
            )
            owner._require_issued_native_plan(plan, budget)
            with pytest.raises(ValueError, match="actual producer-issued full inspected plan"):
                owner._require_issued_native_plan(copy(plan), budget)
            object.__setattr__(plan, "source", copy(source))
            try:
                with pytest.raises(ValueError, match="actual producer-issued full inspected plan"):
                    owner._require_issued_native_plan(plan, budget)
            finally:
                object.__setattr__(plan, "source", source)
            original = source.vertex_residuals
            changed = original.copy()
            changed.flat[0] = np.nextafter(changed.flat[0], np.inf)
            object.__setattr__(source, "vertex_residuals", changed)
            try:
                with pytest.raises(ValueError, match="actual producer-issued full inspected plan"):
                    owner._require_issued_native_plan(plan, budget)
            finally:
                object.__setattr__(source, "vertex_residuals", original)
            owner._require_issued_native_plan(plan, budget)
    identifier = id(plan)
    reference = weakref.ref(plan)
    del plan
    gc.collect()
    assert reference() is None
    assert identifier not in owner._issued_native_plans


def test_plc_scoped_request_is_not_a_root_cad_request(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
) -> None:
    _, surface_source, surface, _ = cylinder_surface
    source = NativeCadVolumeSource(surface_source, surface)
    plc = source.plc_source
    wrong = phx.meshing.MeshingScope(
        plc.source_id,
        plc.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        f"{plc.source_id}:facets",
        np.arange(plc.complex.facet_count, dtype=np.int64),
    )
    with pytest.raises((ValueError, phx.meshing.MeshingFailure)):
        _fill(source, _volume_request(wrong))


def test_coverage_retains_explicit_chain_without_changing_authored_domain(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
) -> None:
    model, surface_source, surface, _ = cylinder_surface
    source = NativeCadVolumeSource(surface_source, surface).fidelity_source
    domain = CadSourceDomain(model, region_ids=("cylinder",))
    identity = domain.domain_id
    assert source.chart_triangulations
    assert _bound_boundary_source(domain, source) is source
    default = _bound_boundary_source(domain, None)
    assert isinstance(default, MeshingDomainBoundarySource)
    assert not default.chart_triangulations
    assert domain.domain_id == identity
    subset = MeshingDomainBoundarySource(
        source.domain,
        (source.patches[0],),
        chart_triangulations=source.chart_triangulations,
    )
    with pytest.raises(ValueError, match="complete authored CAD boundary"):
        _bound_boundary_source(domain, subset)
    foreign = CadSourceDomain(
        phx.geometry.brep_cylinder(
            _RADIUS,
            2.0 * _HEIGHT,
            coordinate_contract=_SI,
            tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
        ),
        region_ids=("cylinder",),
    )
    with pytest.raises(ValueError, match="authority, source revision and trim"):
        _bound_boundary_source(foreign, source)
    changed_domain = eqx.tree_at(
        lambda value: value.domain.corner_points,
        source,
        np.asarray(source.domain.corner_points) + 0.25,
    )
    assert changed_domain.source_id == source.source_id
    assert changed_domain.source_revision == source.source_revision
    with pytest.raises(ValueError, match="authority, source revision and trim"):
        _bound_boundary_source(domain, changed_domain)


@pytest.mark.parametrize("fault", ("gap", "reversed", "foreign_patch"))
def test_analytic_projection_does_not_discard_invalid_explicit_chain(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
    fault: str,
) -> None:
    _, surface_source, surface, _ = cylinder_surface
    source = NativeCadVolumeSource(surface_source, surface).fidelity_source
    rows = list(source.chart_triangulations)
    row = rows[0]
    patch, charts, points, cells = row[:4]
    if fault == "gap":
        cells = cells[1:]
    elif fault == "reversed":
        cells = cells.copy()
        cells[0] = cells[0, [0, 2, 1]]
    else:
        patch = len(source.domain.patches)
    rows[0] = (
        patch, charts, points, cells, row[4], row[5], row[6], row[7], row[8], row[9]
    )
    invalid = MeshingDomainBoundarySource(
        source.domain, source.patches, chart_triangulations=tuple(rows)
    )
    mesh = cylinder_mesh(2, 2)
    certificate = certify_source_fidelity(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        invalid,
        tolerance=_DEVIATION,
    )
    assert certificate.status != "certified"
    assert certificate.boundary_degree_id is None
    assert any(
        finding.check == "source_trim_coverage_premise"
        for finding in certificate.findings
    )


def test_analytic_projection_binds_the_verified_explicit_chart_witness(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
) -> None:
    model, surface_source, surface, _ = cylinder_surface
    source = NativeCadVolumeSource(surface_source, surface).fidelity_source
    domain = CadSourceDomain(model, region_ids=("cylinder",))
    mesh = cylinder_mesh(2, 2)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    explicit = certify_source_fidelity(mesh, geometry, source, tolerance=_DEVIATION)
    authored = certify_source_fidelity(
        mesh, geometry, domain.boundary_source(), tolerance=_DEVIATION
    )
    assert explicit.status == authored.status == "certified"
    assert explicit.boundary_degree_id is not None
    assert authored.boundary_degree_id is not None
    assert explicit.boundary_degree_id != authored.boundary_degree_id
    assert explicit.mesh_to_source_upper <= _DEVIATION
    assert explicit.source_to_mesh_upper <= _DEVIATION


def _bowed_cylinder(offset: float, allowance: float) -> phx.geometry.BRepModel:
    """Keep every authored vertex and face, but bow the top ring inward.

    The ring still starts and ends at its authored vertex, so the domain admits
    it; its interior departs from both incident faces by up to ``2 * offset``.
    """
    model = _cylinder()
    geometry = model.require_geometry()
    rings = [
        (edge, curve)
        for edge, index in enumerate(geometry.edge_curves)
        if isinstance(curve := geometry.curves[index], CircleCurve)
        and float(np.asarray(curve.center)[2]) == _HEIGHT
    ]
    if len(rings) != 1:
        raise ValueError(f"cylinder.top must be one authored ring; got {len(rings)}")
    ((edge, top),) = rings
    ring = CircleCurve(
        np.asarray(top.center) + (offset, 0.0, 0.0),
        np.asarray(top.first_axis),
        np.asarray(top.second_axis),
        float(np.asarray(top.radius)) - offset,
    )

    def rebind(root: object) -> NativePeriodEndpoint | None:
        if root is None:
            return None
        if not isinstance(root, NativePeriodEndpoint):
            raise TypeError(f"cylinder.ring_root must be native; got {type(root).__name__}")
        return NativePeriodEndpoint(
            ring, root.patch, root.axis, rational=root.rational, turns=root.turns
        )

    bowed = BRepGeometry(
        vertex_points=geometry.vertex_points,
        curves=tuple(
            ring if index == geometry.edge_curves[edge] else curve
            for index, curve in enumerate(geometry.curves)
        ),
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        occurrences=geometry.occurrences,
        assembly_containers=geometry.assembly_containers,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=tuple(
            (rebind(row[0]), rebind(row[1])) if index == edge else row
            for index, row in enumerate(geometry.edge_endpoint_roots)
        ),
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
    )
    return model.evolve(
        geometry=bowed,
        report=dataclasses.replace(model.report, curve_surface_tolerance=allowance),
        coedge_deviation_bounds=None,
    )


def test_gapped_capped_cylinder_never_takes_the_zero_gap_analytic_shortcut() -> None:
    """Absent gap tables are not zero: an in-allowance gapped ring is refused."""
    offset, allowance, tolerance = 2.0**-6, 2.0**-2, 0.025
    exact, bowed = _cylinder(), _bowed_cylinder(offset, allowance)
    geometry_ = bowed.require_geometry()
    assert geometry_.edge_face_gap_bounds is None
    assert geometry_.edge_vertex_gap_bounds is None
    # The authored ring is farther than the fidelity request from its faces,
    # yet construction validated it within the declared import allowance.
    assert tolerance < 2.0 * offset <= float(np.max(bowed.coedge_deviation_bounds))
    assert float(np.max(bowed.coedge_deviation_bounds)) <= allowance
    assert complete_cylinder_region(exact) is not None
    assert complete_cylinder_region(bowed) is None
    mesh = cylinder_mesh(3, 2)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    limits = MeshCertificateLimits()
    sources = tuple(
        MeshingDomainBoundarySource(phx.geometry.MeshingDomain.from_brep(model), (0, 1, 2))
        for model in (exact, bowed)
    )
    with coordinate_certificate_request(limits) as budget:
        proofs = tuple(
            analytic_boundary_proof(mesh, geometry, source, tolerance, limits, budget=budget)
            for source in sources
        )
    assert proofs[0] is not None and proofs[0].boundary_degree_id is not None
    assert proofs[0].distance_upper <= tolerance
    assert proofs[1] is None
    certificate = certify_source_fidelity(mesh, geometry, sources[1], tolerance=tolerance)
    assert certificate.status != "certified"
    assert certificate.boundary_degree_id is None
    assert certificate.cad_closure is None


def test_genuine_cad_publication_registration_binds_paid_mutable_payload(
    cylinder_surface: tuple[
        phx.geometry.BRepModel,
        phx.meshing.NativeSurfaceSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.MeshingScope,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing.providers._native_publication import NativePublicationExecution
    from tests._support.coordinate_references import native_meshing_request
    from tools.meshing_qualification.scenarios.cad import (
        _remember_cad_publication_registration,
    )

    _, surface_source, surface, boundary_scope = cylinder_surface
    source = NativeCadVolumeSource(surface_source, surface)
    specification = _volume_request(boundary_scope)
    options = phx.meshing.NativeMeshingOptions("cad_tetrahedral")
    # Planning, actual publication and inspection registration share the
    # original root allowance. This tests live issuance, not a P2 workflow.
    with native_meshing_request(specification.limits) as (scope, budget, workspace):
        plan = phx.meshing.NativeMeshingProvider(options).plan(
            source, specification, coordinate_contract=_SI
        )
        carrier = plan.execute()
        _remember_cad_publication_registration(
            plan, carrier, NativePublicationExecution(scope, workspace, None, budget)
        )
        _assert_issued_registration_payload(
            source, specification, options, carrier, scope, budget, monkeypatch
        )


def _assert_issued_registration_payload(
    source: NativeCadVolumeSource,
    specification: phx.meshing.VolumeMeshingSpec,
    options: phx.meshing.NativeMeshingOptions,
    carrier: phx.meshing.CellMeshingResult,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copy import copy

    from phydrax.lifecycle._meshing_registration import (
        has_issued_native_generation_registration,
    )

    with budget.bind_ticket(scope.ticket_view()):
        before = budget.work_units
        assert has_issued_native_generation_registration(
            source, specification, options, carrier, budget=budget
        )
        assert budget.work_units > before
        assert not has_issued_native_generation_registration(
            source, specification, options, carrier, budget=None
        )
        # Equal portable payloads do not retain live producer authority.
        assert not has_issued_native_generation_registration(
            copy(source), specification, options, carrier, budget=budget
        )
        residuals = source.vertex_residuals
        writeable = residuals.flags.writeable
        original = residuals.flat[0]
        residuals.setflags(write=True)
        try:
            residuals.flat[0] = np.nextafter(original, np.inf)
            assert not has_issued_native_generation_registration(
                source, specification, options, carrier, budget=budget
            )
        finally:
            residuals.flat[0] = original
            residuals.setflags(write=writeable)
        assert has_issued_native_generation_registration(
            source, specification, options, carrier, budget=budget
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                phx.meshing.NativeMeshingProvider, "plan",
                phx.meshing.NativeMeshingProvider.validate,
            )
            assert not has_issued_native_generation_registration(
                source, specification, options, carrier, budget=budget
            )


def test_general_frustum_p2_homotopy_curving_and_manufactured_solve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from time import perf_counter

    from phydrax._meshcore import (
        NativeHostStorageWorkspace,
        NativeParallelExecution,
    )
    from phydrax.lifecycle._meshing_sources import (
        meshing_source_owner_reconstruction_scope,
    )
    from phydrax.meshing.providers._native_publication import (
        bind_native_execution_result,
        NativePublicationExecution,
    )
    from phydrax.meshing.qualification import qualify_hybrid_curved_publication
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._runtime import Deadline, StageLedger
    from tools.meshing_qualification.scenarios.cad import (
        _cad_p2_checkpoint,
        _cad_p2_solve,
        _CadCase,
        _CadVolume,
        _remember_cad_publication_registration,
    )

    controls = QualificationControls()
    deadline = Deadline(perf_counter() + controls.timeout)
    record = {"stages_seconds": {}, "active_phase": "cad_source"}
    ledger = StageLedger(record, deadline=deadline)
    model = ledger.run(
        "cad_source",
        lambda: phx.geometry.brep_cone(
            1.0,
            0.5,
            2.0,
            coordinate_contract=_SI,
            tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
        ),
    )
    limits = phx.meshing.MeshingLimits()
    surface_operands = CoordinateOperandCollector(capture_affine_areas=True)
    with (
        NativeParallelExecution(None, deadline.remaining()),
        NativeExecutionScope(limits.native_limits()) as generation,
    ):
        surface_source, surface, scope = ledger.run(
            "cad_surface", lambda: _surface(model, surface_operands)
        )
        source = NativeCadVolumeSource(surface_source, surface)
        specification = _volume_request(scope)
        options = phx.meshing.NativeMeshingOptions("cad_tetrahedral")

        def generate_volume() -> tuple[
            phx.meshing.NativeMeshingPlan, phx.meshing.CellMeshingResult
        ]:
            plan = phx.meshing.NativeMeshingProvider(options).plan(
                source, specification, coordinate_contract=_SI
            )
            return plan, plan.execute(
                coordinate_operands=CoordinateOperandCollector(
                    source_operands=surface_operands.prepared
                )
            )

        plan, volume = ledger.run("cad_volume", generate_volume)
    assert generation.evidence is not None
    volume = bind_native_execution_result(
        volume, generation.evidence, limits, generation.operation_started
    )
    projection = phx.geometry.prepare_brep_projection(model)
    # Publication retains its certified vertex association, as the CAD
    # qualification caller consumes it; it is not re-derived here.
    vertex_set = volume.mesh.entity_set(0).entity_set_id
    association = next(
        value for value in volume.associations if value.target_entity_set_id == vertex_set
    )
    domain = CadSourceDomain(model, region_ids=("frustum",))
    regions = np.zeros(volume.mesh.blocks[0].cell_count, dtype=np.int64)
    proof_limits = MeshCertificateLimits()
    assert volume.certification is not None
    publication_audit = required_audit_policy(
        phx.meshing.CellMeshAuditPolicy(), volume.certification.schedule.required_audit_checks
    )
    import phydrax.geometry._mapped_embedding as mapped_owner

    def repeated_homotopy(*args: object, **kwargs: object) -> None:
        pytest.fail("publication must consume the actual issued CAD homotopy, not repeat it")

    limits = specification.limits
    with (
        NativeParallelExecution(None, deadline.remaining()),
        NativeExecutionScope(limits.native_limits()) as execution,
        NativeHostStorageWorkspace(execution) as workspace,
    ):
        with execution.batch(0) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
        with (
            budget.retained_storage(lambda size: execution.batch(0, bytes_upper=size)),
            meshing_source_owner_reconstruction_scope(budget=budget),
        ):
            with budget.bind_ticket(execution.ticket_view()):
                curved = ledger.run(
                    "cad_p2_curving",
                    lambda: phx.meshing.curve_cell_mesh(
                        volume.mesh,
                        association,
                        projection,
                        source=source.fidelity_source,
                        domain=domain,
                        cell_regions=regions,
                        policy=phx.meshing.HighOrderCurvingPolicy(
                            degree=2, fidelity_tolerance=_DEVIATION,
                            validity=publication_audit.validity_policy,
                        ),
                        certificate_limits=proof_limits,
                        budget=budget,
                    ),
                )
                assert curved.status is CURVING_STATUS.CURVED
                assert curved.evidence.accepted
                assert curved.evidence.fidelity is not None
                assert curved.evidence.fidelity.boundary_degree_id is not None
                closure = curved.evidence.fidelity.cad_closure
                assert closure is not None
                assert closure.binding.binding_id == curved.evidence.fidelity.binding.binding_id
                assert closure.representative_id != model.source_id
                assert closure.gap_upper <= _DEVIATION
                np.testing.assert_array_equal(
                    closure.gap_bounds,
                    type(closure).from_stored_fields(closure).gap_bounds,
                )
                assert curved.evidence.coverage is not None
                assert curved.evidence.coverage.status == "certified"
                inputs = phx.meshing.MeshCertificationInputs(
                    volume.mesh,
                    curved.geometry,
                    volume.certification.schedule,
                    domain=domain,
                    cell_regions=regions,
                    source=source.fidelity_source,
                    fidelity_tolerance=_DEVIATION,
                    limits=proof_limits,
                    budget=budget,
                )
            with monkeypatch.context() as patch:
                patch.setattr(mapped_owner, "certify_interval_boundary_homotopy", repeated_homotopy)
                carrier = ledger.run(
                    "cad_p2_publication",
                    lambda: qualify_hybrid_curved_publication(
                        volume, curved.geometry, association, inputs,
                        limits=limits, curving=curved.evidence,
                        execution=NativePublicationExecution(execution, workspace, None, budget),
                    ),
                )
            _remember_cad_publication_registration(
                plan, carrier, NativePublicationExecution(execution, workspace, None, budget)
            )
            _assert_issued_registration_payload(
                source, specification, options, carrier, execution, budget, monkeypatch
            )
    assert execution.evidence is not None
    carrier = bind_native_execution_result(
        carrier, execution.evidence, limits, execution.operation_started,
        history_evidence=()
        if volume.execution_evidence is None else (volume.execution_evidence,),
    )
    assert carrier.certification is not None
    assert carrier.certification.embedding is not None
    assert carrier.audit.validity is not None
    assert carrier.audit.validity.policy_id == phx.meshing.CellMeshAuditPolicy().validity_policy.policy_id
    assert carrier.certification.embedding.validity_certificate_id == carrier.audit.validity.certificate_id
    case = _CadCase(
        model=model,
        loops=(),
        reference_volume=7.0 * math.pi / 6.0,
        manufactured_length=1.0,
        size=_SIZE,
        deviation=_DEVIATION,
        minimum_radius=0.5,
    )
    published = _CadVolume(
        domain=surface_source.domain, surface=surface, volume=volume,
        generation_source=source,
        generation_specification=specification,
        generation_options=options,
        generation_plan=plan,
    )
    from phydrax.lifecycle._meshing_field_records import (
        rebind_meshing_accepted_epoch_execution,
    )
    from phydrax.meshing.qualification._native_lifecycle import execute_native_consumer
    from tools.meshing_qualification._runtime import qualification_failure

    consumed = ledger.run(
        "cad_p2_physical_solve",
        lambda: execute_native_consumer(
            carrier, specification.limits,
            lambda budget: _cad_p2_solve(
                carrier, case, controls, deadline=deadline, budget=budget
            ),
            target=lambda value: value[0].carrier,
            failure_factory=qualification_failure,
        ),
    )
    accepted, continued, solved = consumed.value
    accepted = rebind_meshing_accepted_epoch_execution(
        accepted, consumed.result, transition=None
    )
    checkpoint = ledger.run(
        "cad_p2_physical_checkpoint",
        lambda: _cad_p2_checkpoint(
            published, accepted, continued, solved, case, controls, tmp_path, ledger
        ),
    )
    free_dofs, relative_error = solved["free_dofs"], solved["relative_l2_error"]
    assert isinstance(free_dofs, int) and free_dofs > 0
    assert isinstance(relative_error, float) and relative_error <= controls.target_error
    assert checkpoint["accepted_fields_bitwise_preserved"]
    assert checkpoint["fresh_interpreter"]["warm_cold_continuation_bitwise_equal"]
    assert checkpoint["fresh_interpreter"]["source_certification_scientifically_preserved"]
    assert checkpoint["fresh_interpreter"]["cold_request"]["limits_id"] == limits.limits_id
    assert checkpoint["fresh_interpreter"]["cold_request"]["work_units"] > 0
    assert carrier.certification is not None
    assert checkpoint["fresh_interpreter"]["original_certification_id"] == (
        carrier.certification.report_id
    )
    assert checkpoint["fresh_interpreter"]["physical_error"]["relative_l2_error"] <= controls.target_error
    deadline.remaining()
    _assert_cad_boundary_degree_reuse(volume.mesh, curved, source.fidelity_source, monkeypatch)


def _assert_cad_boundary_degree_reuse(
    mesh: phx.discretization.CellMesh,
    curved: phx.meshing.HighOrderCurvingResult,
    source: MeshingDomainBoundarySource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consume a real source proof; no fake issuance or mocked proof creation."""
    from copy import copy

    import phydrax.geometry._boundary_degree_embedding as degree_owner
    import phydrax.geometry._mapped_embedding as mapped_owner
    from phydrax.geometry._global_embedding import certify_global_embedding
    from phydrax.geometry._source_boundary_records import CadBoundaryClosureEvidence
    from phydrax.geometry._source_fidelity import require_issued_cad_boundary_closure
    from phydrax.meshing._certification import require_prepared_embedding

    assert curved.evidence.fidelity is not None
    closure = curved.evidence.fidelity.cad_closure
    assert closure is not None
    limits = MeshCertificateLimits()
    require_issued_cad_boundary_closure(mesh, curved.geometry, closure, limits)
    require_issued_cad_boundary_closure(
        mesh, curved.geometry, closure, limits, source=source
    )
    # Source-aware consumers must authenticate arrays, not asserted source IDs.
    data = source.chart_triangulations[0]
    changed = eqx.tree_at(
        lambda value: value.chart_triangulations[0][3], source, data[3][:-1]
    )
    with pytest.raises(ValueError, match="actual issued source"):
        require_issued_cad_boundary_closure(
            mesh, curved.geometry, closure, limits, source=changed
        )
    original_embedding = curved.evidence.embedding
    assert original_embedding is not None and original_embedding.boundary_degree is not None
    require_prepared_embedding(
        mesh, curved.geometry, curved.evidence.certificate, original_embedding, limits
    )
    missing = copy(original_embedding)
    object.__setattr__(
        missing, "evaluated_checks",
        tuple(value for value in missing.evaluated_checks if value != "mapped_shared_entity_conformity"),
    )
    with pytest.raises(ValueError, match="sufficient governed evidence"):
        require_prepared_embedding(
            mesh, curved.geometry, curved.evidence.certificate, missing, limits
        )
    wrong_degree = copy(original_embedding.boundary_degree)
    object.__setattr__(
        wrong_degree, "conforming_entity_count", wrong_degree.conforming_entity_count + 1
    )
    wrong_counts = copy(original_embedding)
    object.__setattr__(wrong_counts, "boundary_degree", wrong_degree)
    with pytest.raises(ValueError, match="degree counts must match"):
        require_prepared_embedding(
            mesh, curved.geometry, curved.evidence.certificate, wrong_counts, limits
        )

    def duplicate_proof(*args: object, **kwargs: object) -> None:
        pytest.fail("the CAD consumer must not repeat homotopy, local univalence or boundary contact")

    with monkeypatch.context() as patch:
        patch.setattr(mapped_owner, "certify_interval_boundary_homotopy", duplicate_proof)
        patch.setattr(mapped_owner, "root_injective", duplicate_proof)
        patch.setattr(degree_owner, "_boundary_injectivity", duplicate_proof)
        # A fresh request cannot satisfy this test through embedding's cache.
        embedding = certify_global_embedding(
            mesh, curved.geometry, curved.evidence.certificate,
            limits=limits, cad_boundary=closure,
        )
    assert embedding.status == "certified"
    assert {
        "cell_validity", "facet_pairing", "mapped_trace_continuity",
        "mapped_closed_cell_orientation", "mapped_shared_entity_conformity",
        "mapped_boundary_injectivity", "mapped_shell_exterior_degree",
    } <= set(embedding.evaluated_checks)
    assert embedding.cad_boundary_closure is closure
    assert "mapped_cad_boundary_homotopy_reuse" in embedding.evaluated_checks
    assert "mapped_local_injectivity" not in embedding.evaluated_checks
    assert embedding.boundary_degree is not None
    assert embedding.boundary_degree.status == "embedded"
    assert embedding.boundary_degree.maximum_degree == 1
    assert not any(embedding.boundary_degree.shell_exterior_degrees)
    assert embedding.boundary_degree.boundary_pair_count == 0
    assert closure.homotopy_piece_count > 0
    assert closure.homotopy_distance_upper <= _DEVIATION

    def rehashed_report(partial: bool, homotopy: str) -> CadBoundaryClosureEvidence:
        selected = slice(None, -1) if partial else slice(None)
        return CadBoundaryClosureEvidence(
            closure.binding,
            authority_id=closure.authority_id,
            source_scope_id=closure.source_scope_id,
            representative_id=closure.representative_id,
            homotopy_id=homotopy,
            homotopy_distance_upper=closure.homotopy_distance_upper,
            homotopy_piece_count=closure.homotopy_piece_count,
            chart_indices=closure.chart_indices[selected],
            target_facet_ids=closure.target_facet_ids[selected],
            gap_bounds=closure.gap_bounds[selected],
            original_gap_allowance=closure.original_gap_allowance,
        )

    # Equal archival/constructor reports and rehashed partial/forged reports
    # are not independently earned capabilities, even with certified fidelity.
    for report in (
        CadBoundaryClosureEvidence.from_stored_fields(closure),
        rehashed_report(True, closure.homotopy_id),
        rehashed_report(False, "forged-whole-boundary-homotopy"),
    ):
        with pytest.raises(ValueError, match="independently issued"):
            certify_global_embedding(
                mesh, curved.geometry, curved.evidence.certificate,
                limits=limits, cad_boundary=report,
            )
    stale = eqx.tree_at(
        lambda value: value.gap_bounds, closure,
        np.nextafter(closure.gap_bounds, np.inf),
    )
    with pytest.raises(ValueError, match="CadBoundaryClosureEvidence"):
        require_issued_cad_boundary_closure(mesh, curved.geometry, stale, limits)
    foreign_mesh = eqx.tree_at(lambda value: value.coordinates, mesh, mesh.coordinates)
    with pytest.raises(ValueError, match="independently issued"):
        require_issued_cad_boundary_closure(foreign_mesh, curved.geometry, closure, limits)
    changed_geometry = eqx.tree_at(
        lambda value: value.coordinates, curved.geometry,
        curved.geometry.coordinates + 0.125,
    )
    with pytest.raises(ValueError, match="MeshCertificateBinding"):
        require_issued_cad_boundary_closure(mesh, changed_geometry, closure, limits)
    changed_limits = MeshCertificateLimits(maximum_ray_tests=limits.maximum_ray_tests + 1)
    with pytest.raises(ValueError, match="independently issued"):
        require_issued_cad_boundary_closure(mesh, curved.geometry, closure, changed_limits)


def test_combined_cad_request_keeps_surface_operands_paid_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._meshcore import current_native_execution_scope
    from phydrax.discretization._coordinate_operands import PreparedCoordinateOperands
    from phydrax.geometry import _affine_chart_coverage as affine_owner
    from phydrax.meshing._native_preparation import native_preparation_ledger
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios.cad import _cad_volume_mesh, _CadCase

    model = phx.geometry.brep_cone(
        _RADIUS, _RADIUS / 2.0, _HEIGHT, coordinate_contract=_SI,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    case = _CadCase(
        model=model, loops=(), reference_volume=7.0 * math.pi * _RADIUS**2 * _HEIGHT / 12.0,
        manufactured_length=_HEIGHT, size=_SIZE, deviation=_DEVIATION,
        minimum_radius=_RADIUS,
    )
    reused: list[PreparedCoordinateOperands] = []
    original = PreparedCoordinateOperands.require
    area_hits: list[tuple[affine_owner._Triangle, affine_owner._Triangle]] = []
    original_area = affine_owner._Preparation.area

    def paid_area(
        preparation: affine_owner._Preparation,
        first: affine_owner._Triangle,
        second: affine_owner._Triangle,
    ) -> Fraction:
        value = original_area(preparation, first, second)
        if reused and (first, second) in preparation.known_areas:
            area_hits.append((first, second))
            assert value == preparation.known_areas[first, second]
        return value

    monkeypatch.setattr(affine_owner._Preparation, "area", paid_area)

    def require_paid(
        self: PreparedCoordinateOperands,
        mesh: phx.discretization.CellMesh,
        geometry: phx.discretization.CellGeometrySpec,
        receipt: phx.meshing.NativeExecutionRecord,
        budget: CoordinateEnclosureBudget,
        *,
        coordinate_contract: phx.SpatialCoordinateContract,
    ) -> None:
        execution = current_native_execution_scope()
        assert execution is not None and execution.parent is not None
        assert native_preparation_ledger(execution).receipt_is_accounted(
            execution, self.earning_receipt
        )
        assert self._affine_areas
        original(
            self, mesh, geometry, receipt, budget,
            coordinate_contract=coordinate_contract,
        )
        assert all(budget.affine_chain_intersection_cache[key] == value
                   for key, value in self._affine_areas)
        forged = dataclasses.replace(self, _affine_areas=())
        with pytest.raises(ValueError, match="bind the current source"):
            original(
                forged, mesh, geometry, receipt, budget,
                coordinate_contract=coordinate_contract,
            )
        key, area = self._affine_areas[0]
        budget.affine_chain_intersection_cache[key] = area + 1
        try:
            with pytest.raises(ValueError, match="conflict"):
                original(
                    self, mesh, geometry, receipt, budget,
                    coordinate_contract=coordinate_contract,
                )
        finally:
            budget.affine_chain_intersection_cache[key] = area
        reused.append(self)

    monkeypatch.setattr(PreparedCoordinateOperands, "require", require_paid)
    published = _cad_volume_mesh(
        model, case, QualificationControls(), lambda phase: None
    )
    assert len(reused) == 1
    assert area_hits
    receipt = published.volume.execution_evidence
    surface_receipt = published.surface.execution_evidence
    assert receipt is not None and surface_receipt is not None
    receipt.require_valid()
    assert receipt.preparation_evidence is None
    assert int(np.asarray(receipt.total_work_units)) == int(
        np.asarray(receipt.work_counters.construction_work)
    )
    assert int(np.asarray(receipt.total_work_units)) > int(
        np.asarray(surface_receipt.total_work_units)
    )
    assert not receipt.history_evidence
    monkeypatch.setattr(PreparedCoordinateOperands, "require", original)
    # A completed public surface cannot acquire a paid phase in another request
    # merely because its result, geometry, IDs and exact triangle pairs match.
    with NativeExecutionScope(published.generation_specification.limits.native_limits()) as cold:
        with cold.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=published.generation_specification.limits.maximum_work_units,
                maximum_memory_bytes=published.generation_specification.limits.maximum_scratch_bytes,
            )
            with pytest.raises(ValueError, match="already be paid"):
                reused[0].require(
                    published.surface.mesh, published.surface.geometry,
                    surface_receipt, budget,
                    coordinate_contract=published.surface.coordinate_contract,
                )
    import gc
    import pickle
    import weakref

    with pytest.raises(TypeError, match="archive/cold"):
        pickle.dumps(reused[0])
    issued = weakref.ref(reused[0])
    reused.clear()
    gc.collect()
    assert issued() is None


def test_cad_domain_integrity_uses_canonical_request_owner_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        meshing_source_owner_reconstruction_scope,
    )
    from phydrax.meshing._certification_inputs import _validate_domain_integrity

    model = _cylinder()
    domain = CadSourceDomain(model, region_ids=("cylinder",))
    bank = np.array(model.require_geometry().vertex_points, copy=True)
    bank.setflags(write=False)
    stored = domain
    original = CadSourceDomain.from_stored_fields
    calls: list[CoordinateEnclosureBudget | None] = []

    def reconstruct(
        cls: type[CadSourceDomain], value: CadSourceDomain, /, *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> CadSourceDomain:
        calls.append(budget)
        return original(value, budget=budget)

    monkeypatch.setattr(CadSourceDomain, "from_stored_fields", classmethod(reconstruct))
    geometry_type = type(model.require_geometry())
    geometry_constructor = geometry_type.from_stored_fields
    geometry_budgets: list[CoordinateEnclosureBudget | None] = []
    restored_geometry = []

    def reconstruct_geometry(
        cls: type[BRepGeometry], value: BRepGeometry, /, *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> BRepGeometry:
        geometry_budgets.append(budget)
        fresh = geometry_constructor(value, budget=budget)
        restored_geometry.append(fresh)
        return fresh

    monkeypatch.setattr(geometry_type, "from_stored_fields", classmethod(reconstruct_geometry))
    limits = phx.meshing.MeshingLimits()
    with meshing_source_owner_reconstruction_scope():
        with NativeExecutionScope(limits.native_limits()) as first:
            with first.ticket_view() as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket, maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                with (
                    meshing_source_owner_reconstruction_scope(budget=budget),
                ):
                    _validate_domain_integrity(stored, budget=budget)
                    assert calls == [budget]
                    assert restored_geometry and restored_geometry[-1] is not stored.model.geometry
                    np.testing.assert_array_equal(restored_geometry[-1].vertex_points, bank)
                    _validate_domain_integrity(stored, budget=budget)
                    assert calls == [budget]
                    previous = float(bank[0, 0])
                    bank.setflags(write=True)
                    bank[0, 0] = previous + 0.125
                    bank.setflags(write=False)
                    changed = eqx.tree_at(
                        lambda value: value.model.geometry.vertex_points, stored, jnp.asarray(bank)
                    )
                    try:
                        with pytest.raises(ValueError):
                            type(model).from_stored_fields(changed.model, budget=budget)
                        with pytest.raises(ValueError):
                            _validate_domain_integrity(changed, budget=budget)
                    finally:
                        bank.setflags(write=True)
                        bank[0, 0] = previous
                        bank.setflags(write=False)
                    assert len(calls) == 2
                    _validate_domain_integrity(stored, budget=budget)
                    assert len(calls) == 2

                    def replaced_constructor(
                        cls: type[CadSourceDomain], value: CadSourceDomain, /, *,
                        budget: CoordinateEnclosureBudget | None = None,
                    ) -> CadSourceDomain:
                        calls.append(budget)
                        return original(value, budget=budget)

                    monkeypatch.setattr(
                        CadSourceDomain, "from_stored_fields", classmethod(replaced_constructor)
                    )
                    _validate_domain_integrity(stored, budget=budget)
                    assert len(calls) == 3
                    model_constructor = type(model).from_stored_fields
                    nested_calls: list[CoordinateEnclosureBudget | None] = []

                    def replaced_model_constructor(
                        cls: type[phx.geometry.BRepModel], value: phx.geometry.BRepModel, /, *,
                        budget: CoordinateEnclosureBudget | None = None,
                    ) -> phx.geometry.BRepModel:
                        nested_calls.append(budget)
                        return model_constructor(value, budget=budget)

                    monkeypatch.setattr(
                        type(model), "from_stored_fields", classmethod(replaced_model_constructor)
                    )
                    _validate_domain_integrity(stored, budget=budget)
                    assert len(calls) == 4 and nested_calls
                    nested_count = len(nested_calls)
        with NativeExecutionScope(limits.native_limits()) as second:
            with (
                budget.bind_ticket(second.ticket_view()),
                meshing_source_owner_reconstruction_scope(budget=budget),
            ):
                _validate_domain_integrity(stored, budget=budget)
                assert len(calls) == 5 and len(nested_calls) > nested_count
                _validate_domain_integrity(stored, budget=budget)
                assert len(calls) == 5
        assert all(actual is budget for actual in (*calls, *nested_calls, *geometry_budgets))


def test_brep_model_restore_revalidates_nested_geometry_on_actual_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _cylinder()
    bank = np.array(model.require_geometry().vertex_points, copy=True)
    bank.setflags(write=False)
    stored = eqx.tree_at(lambda value: value.geometry.vertex_points, model, bank)
    geometry_type = type(model.require_geometry())
    original = geometry_type.from_stored_fields
    paid = []

    def restore_geometry(
        cls: type[BRepGeometry], value: BRepGeometry, /, *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> BRepGeometry:
        paid.append(budget)
        return original(value, budget=budget)

    monkeypatch.setattr(geometry_type, "from_stored_fields", classmethod(restore_geometry))
    limits = phx.meshing.MeshingLimits()
    with NativeExecutionScope(limits.native_limits()) as execution:
        with execution.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            restored = type(model).from_stored_fields(model, budget=budget)
            assert restored.geometry is not model.geometry
            assert restored.model_id == stored.model_id
            np.testing.assert_array_equal(restored.require_geometry().vertex_points, bank)
            bank.setflags(write=True)
            bank[0, 0] += 0.125
            bank.setflags(write=False)
            with pytest.raises(ValueError):
                type(model).from_stored_fields(stored, budget=budget)
            changed_jax = eqx.tree_at(
                lambda value: value.geometry.vertex_points, model, jnp.asarray(bank)
            )
            with pytest.raises(ValueError):
                type(model).from_stored_fields(changed_jax, budget=budget)
            assert len(paid) == 3 and all(current is budget for current in paid)

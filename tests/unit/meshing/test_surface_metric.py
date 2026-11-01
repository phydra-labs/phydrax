#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._geometry_predicates import PredicateMode
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._surface_chart_records import SurfaceChartWitness
from phydrax.geometry import Sphere
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry._surface_source_support import SurfaceSourceCharts
from phydrax.geometry.brep._constructors import brep_sphere, brep_torus
from phydrax.geometry.brep._patches import LineCurve, PlanePatch
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.meshing._adaptation import (
    execute_mesh_adaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._implicit_association_transfer import ImplicitAssociationTransfer
from phydrax.meshing._metric import MeshMetricField, metric_edge_lengths
from phydrax.meshing._periodic import periodic_entity_orbit_map
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing._surface_association_transfer import SurfaceAssociationTransfer
from phydrax.meshing._surface_metric import (
    execute_surface_metric_adaptation,
    prepare_implicit_surface_metric_source,
    prepare_surface_metric_source,
    SurfaceMetricOutcome,
    SurfaceMetricStatus,
)
from phydrax.meshing._topology_edit import (
    assemble_topology_edit,
    entity_keys,
    TopologyEditBlock,
)
from phydrax.meshing.providers._native import NativeMeshingProvider
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_schedules import NativeSurfaceSchedule
from phydrax.meshing.providers._native_sources import NativeImplicitSource
from phydrax.meshing.providers._native_surface import prepare_native_surface


@pytest.fixture(scope="module")
def _trimmed_sheet() -> tuple[MeshingDomain, CellMeshingResult]:
    contract = SpatialCoordinateContract.si()
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    curves = tuple(MeshingDomainCurve(row, (row + 1) % 4) for row in range(4))
    uses = tuple(
        PatchCurveUse(
            row,
            LineCurve(
                np.asarray(corners[row], dtype=np.float64),
                np.asarray(corners[(row + 1) % 4], dtype=np.float64)
                - np.asarray(corners[row], dtype=np.float64),
            ),
            0.0,
            1.0,
            sense=1,
        )
        for row in range(4)
    )
    domain = MeshingDomain(
        (
            MeshingSurfacePatch(
                PlanePatch(
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
                ((uses),),
            ),
        ),
        curves,
        4,
        source_id="surface-metric-trimmed-sheet",
        source_revision="surface-metric-sheet-r1",
        regions=(MeshingDomainRegion("sheet", ((0, 1),)),),
    )
    scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.8, strength=SizeControlStrength.HARD),
        ),
        maximum_deviation=0.015,
    )
    options = NativeMeshingOptions(
        "parametric_surface",
        surface_schedule=NativeSurfaceSchedule(spacing_fraction=0.45),
    )
    source = (
        NativeMeshingProvider(options)
        .plan(domain, specification, coordinate_contract=contract)
        .execute()
    )
    if source.surface_source is None:
        raise AssertionError("Native trimmed-sheet acceptance lost its source charts.")
    return domain, source


def _closed_brep_surface(
    family: str, /
) -> tuple[MeshingDomain, CellMesh, SurfaceSourceCharts]:
    contract = SpatialCoordinateContract.si()
    if family == "sphere":
        model = brep_sphere(
            0.75,
            coordinate_contract=contract,
            source_id="surface-metric-brep-sphere",
        )
    elif family == "torus":
        model = brep_torus(
            0.9,
            0.3,
            coordinate_contract=contract,
            source_id="surface-metric-brep-torus",
        )
    else:
        raise ValueError(f"Unknown BRep source family {family!r}.")
    domain = MeshingDomain.from_brep(model)
    scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 1.2, strength=SizeControlStrength.HARD),
        ),
        maximum_deviation=0.1,
    )
    construction = prepare_native_surface(
        domain,
        specification,
        NativeMeshingOptions(
            "parametric_surface",
            surface_schedule=NativeSurfaceSchedule(spacing_fraction=0.65),
        ),
        contract,
    ).execute()
    source_charts = construction.organization.surface_source
    if source_charts is None:
        raise AssertionError("Closed BRep construction lost its exact source charts.")
    return domain, construction.mesh, source_charts


@pytest.mark.parametrize("family", ("sphere", "torus"), ids=("brep-sphere", "brep-torus"))
def test_closed_brep_source_chart_metric_adaptation(family: str) -> None:
    domain, mesh, source_charts = _closed_brep_surface(family)
    # Curvature-rate trim sampling leaves no source edge longer than sqrt(2) in
    # the base metric; doubling it requests refinement on every family.
    prepared = prepare_surface_metric_source(
        mesh, source_charts, 2.0 * _anisotropic_metric(mesh)
    )
    outcome = execute_surface_metric_adaptation(
        prepared,
        predicate_mode=PredicateMode.EXACT,
        topology_operations=True,
        relocation=True,
    )
    assert outcome.splits + outcome.collapses + outcome.flips + outcome.relocations > 0
    assert outcome.status is not SurfaceMetricStatus.NUMERICAL_STOP
    _exact_target_points(outcome, domain)
    target_mesh, _, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="surface-metric-closed-brep"
    )
    mapped_cells = np.concatenate(
        tuple(
            np.asarray(group.target_cell_ids, dtype=np.int64)
            for group in outcome.geometry.reference_maps
        )
    )
    target_cell_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks
        )
    )
    np.testing.assert_array_equal(np.sort(mapped_cells), np.sort(target_cell_ids))


def _implicit_sphere_source(
    maximum_deviation: float = 0.5,
) -> tuple[AnalyticImplicitProfile, CellMeshingResult, SpatialCoordinateContract]:
    contract = SpatialCoordinateContract.si()
    source_id = "surface-metric-implicit-sphere"
    revision = f"{source_id}-r1"
    geometry = Sphere((0.0, 0.0, 0.0), 0.75, feature_id=source_id).compile()
    profile = AnalyticImplicitProfile(
        geometry,
        contract,
        source_id=source_id,
        source_revision=revision,
    )
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.4,
        root_tolerance=1.0e-8,
    )
    bounds = np.asarray(profile.boundary_bounds, dtype=np.float64)
    padding = float(np.max(bounds[1] - bounds[0])) / 4.0
    discovery_domain = np.stack(
        (
            np.nextafter(bounds[0] - padding, -np.inf),
            np.nextafter(bounds[1] + padding, np.inf),
        )
    )
    discovery = discover_adaptive_implicit_surface(
        profile.geometry,
        domain=discovery_domain,
        policy=policy,
        source_id=profile.source_id,
    )
    native_source = NativeImplicitSource(
        profile.geometry,
        profile.source_id,
        profile.source_revision,
        discovery=discovery,
        authoring_profile=profile,
    )
    scope = MeshingScope(
        profile.source_id,
        profile.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        profile.scope_entity_set_id(2),
        profile.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(),
        maximum_deviation=maximum_deviation,
    )
    options = NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=policy,
    )
    source = (
        NativeMeshingProvider(options)
        .plan(native_source, specification, coordinate_contract=contract)
        .execute()
    )
    return profile, source, contract


def _anisotropic_metric(mesh: CellMesh, /) -> np.ndarray:
    tensor = np.diag(np.asarray((16.0, 20.0, 24.0), dtype=np.float64))
    return np.broadcast_to(tensor, (mesh.coordinates.shape[0], 3, 3)).copy()


def _exact_target_points(
    outcome: SurfaceMetricOutcome,
    domain: MeshingDomain,
    /,
) -> None:
    edit = outcome.edit
    witness = outcome.geometry.target_chart_witness
    if not isinstance(witness, SurfaceChartWitness):
        raise TypeError("The trimmed sheet test requires a SurfaceChartWitness.")
    ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    patches = np.asarray(witness.patches, dtype=np.int64)
    blocks = {
        int(identifier): block
        for block in edit.blocks
        if isinstance(block, TopologyEditBlock)
        for identifier in block.cell_ids.tolist()
    }
    for row, cell_id in enumerate(ids.tolist()):
        chart = witness.exact_chart(row)
        block = blocks[int(cell_id)]
        cell_row = int(np.flatnonzero(np.asarray(block.cell_ids) == cell_id)[0])
        cell_vertices = np.asarray(block.cells[cell_row], dtype=np.int64)
        parameters = np.asarray(
            [[float(value) for value in point] for point in chart],
            dtype=np.float64,
        )
        source_points = domain.evaluate(
            np.full((3,), patches[row], dtype=np.int64), parameters
        )
        target_points = np.asarray(edit.coordinates)[cell_vertices]
        np.testing.assert_allclose(target_points, source_points, rtol=0.0, atol=2.0e-13)


@pytest.mark.strict_jax
def test_open_trimmed_sheet_metric_statistics_match_independent_pullback(
    _trimmed_sheet: tuple[MeshingDomain, CellMeshingResult],
) -> None:
    domain, source = _trimmed_sheet
    mesh = source.mesh
    source_charts = source.surface_source
    if source_charts is None:
        raise AssertionError("Accepted trimmed sheet lost its source charts.")
    metric = np.broadcast_to(
        np.diag(np.asarray((4.0, 1.0, 3.0), dtype=np.float64)),
        (mesh.coordinates.shape[0], 3, 3),
    ).copy()
    prepared = prepare_surface_metric_source(mesh, source_charts, metric)
    outcome = execute_surface_metric_adaptation(
        prepared,
        predicate_mode=PredicateMode.EXACT,
        topology_operations=False,
        relocation=False,
    )
    surface = domain.patches[0].surface
    differential = jax.jacfwd(surface.evaluate)
    charts = jnp.asarray(
        [[float(value) for value in point] for point in prepared.work_vertex_charts],
        dtype=jnp.float64,
    )
    jacobians = np.asarray(jax.vmap(differential)(charts), dtype=np.float64)
    independent = np.stack(
        [
            jacobian.T @ metric[int(source_row)] @ jacobian
            for jacobian, source_row in zip(
                jacobians, prepared.work_to_source_rows, strict=True
            )
        ]
    )
    np.testing.assert_allclose(
        prepared.pullback_metric, independent, rtol=2.0e-13, atol=2.0e-13
    )
    work_vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(prepared.work_mesh.vertex_global_ids, dtype=np.int64).tolist()
        )
    }
    edge_ids = np.asarray(entity_keys(prepared.work_mesh, 1), dtype=np.int64)
    edges = np.asarray(
        [[work_vertex_rows[int(identifier)] for identifier in edge] for edge in edge_ids],
        dtype=np.int64,
    )
    lengths = np.asarray(
        metric_edge_lengths(
            independent, np.asarray(prepared.work_mesh.coordinates), edges
        ),
        dtype=np.float64,
    )
    measured = lengths[~prepared.source_protected_edges]
    lower, upper = 1.0 / np.sqrt(2.0), np.sqrt(2.0)
    assert outcome.evidence.minimum_metric_length == pytest.approx(
        float(np.min(measured))
    )
    assert outcome.evidence.maximum_metric_length == pytest.approx(
        float(np.max(measured))
    )
    assert outcome.evidence.unit_fraction == pytest.approx(
        float(np.mean((measured >= lower) & (measured <= upper)))
    )
    assert outcome.status is SurfaceMetricStatus.COMPLETE


def test_open_trimmed_sheet_adaptation_keeps_boundary_and_exact_source_points(
    _trimmed_sheet: tuple[MeshingDomain, CellMeshingResult],
) -> None:
    domain, source = _trimmed_sheet
    mesh = source.mesh
    source_charts = source.surface_source
    if source_charts is None:
        raise AssertionError("Accepted trimmed sheet lost its source charts.")
    metric = _anisotropic_metric(mesh)
    prepared = prepare_surface_metric_source(mesh, source_charts, metric)
    outcome = execute_surface_metric_adaptation(
        prepared,
        predicate_mode=PredicateMode.EXACT,
        topology_operations=True,
        relocation=True,
    )
    assert outcome.splits + outcome.collapses + outcome.flips + outcome.relocations > 0
    assert outcome.status is not SurfaceMetricStatus.NUMERICAL_STOP
    assert outcome.geometry.reference_maps
    target_mesh, _, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="surface-metric-test"
    )
    mapped_cells = np.concatenate(
        tuple(
            np.asarray(group.target_cell_ids, dtype=np.int64)
            for group in outcome.geometry.reference_maps
        )
    )
    target_cell_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks
        )
    )
    np.testing.assert_array_equal(np.sort(mapped_cells), np.sort(target_cell_ids))
    _exact_target_points(outcome, domain)
    original_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    boundary_edges = np.flatnonzero(prepared.source_protected_edges)
    if boundary_edges.size:
        edge_keys = np.asarray(entity_keys(mesh, 1), dtype=np.int64)[boundary_edges]
        boundary_ids = np.unique(edge_keys)
        source_rows = {
            int(identifier): row
            for row, identifier in enumerate(original_vertex_ids.tolist())
        }
        target_rows = {
            int(identifier): row
            for row, identifier in enumerate(
                np.asarray(outcome.edit.vertex_global_ids, dtype=np.int64).tolist()
            )
        }
        for identifier in boundary_ids.tolist():
            np.testing.assert_array_equal(
                np.asarray(outcome.edit.coordinates)[target_rows[int(identifier)]],
                np.asarray(mesh.coordinates)[source_rows[int(identifier)]],
            )


def _polar_cap_refinement_metric(mesh: CellMesh, /) -> np.ndarray:
    # The cap z > 0.5 of the radius-0.75 sphere holds the north pole fan and crosses
    # the longitude seam; quadrupling the tensor halves its requested edge length
    # there, while the rest of the sphere asks for edges longer than the extraction's.
    values = _anisotropic_metric(mesh)
    values[np.asarray(mesh.coordinates, dtype=np.float64)[:, 2] > 0.5] *= 4.0
    return values


def _coarsening_metric(mesh: CellMesh, /) -> np.ndarray:
    # An isotropic size of 1/sqrt(30), about 0.18, lies above most extraction edges
    # (median 0.15, shortest 0.016), so the extraction's short edges must collapse.
    tensor = np.diag(np.full((3,), 30.0, dtype=np.float64))
    return np.broadcast_to(tensor, (mesh.coordinates.shape[0], 3, 3)).copy()


def test_implicit_sphere_metric_edits_retain_projective_chart_and_orbits() -> None:
    profile, source, contract = _implicit_sphere_source()
    mesh = source.mesh
    certification = source.certification
    if certification is None:
        raise AssertionError("Implicit sphere source lost its mesh certification.")
    witness = certification.inputs.fidelity_witness
    fidelity_source = certification.inputs.source
    if not isinstance(witness, SourceBoundaryChartWitness) or fidelity_source is None:
        raise AssertionError(
            "Implicit sphere source lost its exact chart witness or query."
        )
    prepared = prepare_implicit_surface_metric_source(
        mesh,
        source.geometry,
        contract,
        profile,
        witness,
        fidelity_source,
        _polar_cap_refinement_metric(mesh),
    )
    # The longitude seam stays an exact lattice quotient; the poles, where longitude
    # is undefined, are the declared collapsed chart boundary.
    assert prepared.work_mesh.periodic_topology is not None
    vertex_orbits = np.asarray(
        periodic_entity_orbit_map(prepared.work_mesh, 0).lifted_to_quotient,
        dtype=np.int64,
    )
    assert np.unique(vertex_orbits).size < vertex_orbits.size
    relative = np.asarray(mesh.coordinates, dtype=np.float64) - np.asarray(
        profile.center, dtype=np.float64
    )
    poles = np.flatnonzero((relative[:, 0] == 0.0) & (relative[:, 1] == 0.0))
    assert poles.size == 2
    np.testing.assert_array_equal(
        np.flatnonzero(prepared.collapsed_source_vertices), poles
    )
    outcome = execute_surface_metric_adaptation(
        prepared,
        predicate_mode=PredicateMode.EXACT,
        topology_operations=True,
        relocation=True,
    )
    # Extraction vertices are not features: the cap refines across the seam and the
    # pole fan while the coarser remainder collapses and relocates free vertices.
    # Regular passes stall on one short birth edge whose endpoint collapses both
    # create an edge above sqrt(2); the stall escape merges its ends at the edge
    # point that minimizes the longest star edge, and the unit mesh is reached.
    assert outcome.status is SurfaceMetricStatus.COMPLETE
    assert outcome.out_of_range_edges == 0
    assert not outcome.evidence.stalled_edges
    assert outcome.splits > 0
    assert outcome.collapses > 0
    assert outcome.relocations > 0
    target_witness = outcome.geometry.target_chart_witness
    if not isinstance(target_witness, SourceBoundaryChartWitness):
        raise AssertionError("Implicit metric adaptation lost its source-bound witness.")
    if (
        outcome.geometry.implicit_cell_keys is None
        or outcome.geometry.implicit_root_edge_keys is None
    ):
        raise AssertionError(
            "Implicit metric adaptation lost root cell or edge ancestry."
        )
    standard_chart = ((0, 0), (1, 0), (0, 1))
    assert all(
        target_witness.source_parameter_chart(row) == standard_chart
        for row in range(len(target_witness.source_chart_ids))
    )
    target_mesh, _, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="surface-metric"
    )
    target_geometry = CellGeometrySpec.affine(target_mesh)
    target_witness.require_current(target_mesh, target_geometry, fidelity_source)
    assert outcome.geometry.periodic_topology_id is not None
    assert outcome.geometry.periodic_edge_orbits > 0
    target_points = np.asarray(outcome.edit.coordinates, dtype=np.float64)
    entity_id = witness.source_entity_ids[0]
    projected = profile.project_boundary(
        target_points,
        entity_ids=(entity_id,) * target_points.shape[0],
        occurrence_paths=(witness.occurrence_paths[0],) * target_points.shape[0],
    )
    assert np.all(np.asarray(projected.resolved, dtype=np.bool_))
    np.testing.assert_allclose(
        np.asarray(projected.points, dtype=np.float64),
        target_points,
        rtol=0.0,
        atol=2.0e-13,
    )


@pytest.mark.parametrize(
    "size_tensor",
    (16.0, 25.0, 49.0, 81.0),
    ids=("size-1/4", "size-1/5", "size-1/7", "size-1/9"),
)
def test_implicit_sphere_pole_fans_reach_unit_mesh_across_scales(
    size_tensor: float,
) -> None:
    # Isotropic sizes 1/4 to 1/9 straddle the extraction's median edge (0.15): the
    # coarse sizes collapse the rings around both poles and 1/9 refines them. Every
    # UV copy of a pole is one physical vertex, so the ring vertices stay free and
    # the unit-mesh criterion is met everywhere, the pole fans included.
    profile, source, contract = _implicit_sphere_source()
    certification = source.certification
    if certification is None:
        raise AssertionError("Implicit sphere source lost its mesh certification.")
    witness = certification.inputs.fidelity_witness
    fidelity_source = certification.inputs.source
    if not isinstance(witness, SourceBoundaryChartWitness) or fidelity_source is None:
        raise AssertionError(
            "Implicit sphere source lost its exact chart witness or query."
        )
    tensor = np.diag(np.full((3,), size_tensor, dtype=np.float64))
    prepared = prepare_implicit_surface_metric_source(
        source.mesh,
        source.geometry,
        contract,
        profile,
        witness,
        fidelity_source,
        np.broadcast_to(tensor, (source.mesh.coordinates.shape[0], 3, 3)).copy(),
    )
    outcome = execute_surface_metric_adaptation(
        prepared,
        predicate_mode=PredicateMode.EXACT,
        topology_operations=True,
        relocation=True,
    )
    assert outcome.status is SurfaceMetricStatus.COMPLETE
    assert outcome.out_of_range_edges == 0
    assert not outcome.evidence.stalled_edges


@pytest.mark.parametrize("maximum_deviation", (0.5, 0.2), ids=("coarse", "fine"))
def test_native_surface_metric_route_reconstructs_and_transfers_implicit_sphere(
    maximum_deviation: float,
) -> None:
    profile, source, _ = _implicit_sphere_source(maximum_deviation)
    transfer = ImplicitAssociationTransfer(profile)
    transfer.validate_source(source)
    values = _coarsening_metric(source.mesh)
    vertex_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    order = np.argsort(vertex_ids, kind="stable")
    vertex_set = source.mesh.entity_set(0)
    scope = MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        vertex_set.entity_set_id,
        vertex_ids[order],
    )
    metric = MeshMetricField(
        scope,
        values[order],
        minimum_size=0.1,
        maximum_size=0.6,
        maximum_anisotropy=2.0,
    )
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_METRIC_SURFACE,
        association_transfer=transfer,
        relocation=True,
    )
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MetricMeshAdaptation(metric),
            policy=policy,
        )
    )
    assert result.route is MeshAdaptationRoute.NATIVE_METRIC_SURFACE
    assert isinstance(result.evidence, SurfaceMetricOutcome)
    # Only declared features anchor the implicit chart, so the coarsening request
    # collapses the extraction's short edges and completes.
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.evidence.out_of_range_edges == 0
    assert result.evidence.collapses > 0
    assert result.transition is not None
    assert result.transition.geometry_transition is not None
    # Target cells overlap several source cells, so vertex data move through the
    # general vertex-stencil transfer: exact on constants, not on ambient linear
    # functions, since target vertices are exact source projections.
    field_transfer = result.transfer
    assert field_transfer is not None
    assert field_transfer.preserves_constants
    assert not field_transfer.preserves_linear
    source_points = np.asarray(source.mesh.coordinates, dtype=np.float64)
    target_points = np.asarray(result.target.mesh.coordinates, dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(field_transfer.apply(np.ones((source_points.shape[0],)))),
        np.ones((target_points.shape[0],)),
        rtol=0.0,
        atol=1.0e-13,
    )
    assert result.surface_chart_witness is not None
    if not isinstance(result.surface_chart_witness, SourceBoundaryChartWitness):
        raise AssertionError(
            "Implicit metric finalization lost its exact source chart witness."
        )
    target_certification = result.target.certification
    if target_certification is None:
        raise AssertionError("Implicit metric finalization lost source certification.")
    target_source_query = target_certification.inputs.source
    if target_source_query is None:
        raise AssertionError("Implicit metric finalization lost its source query.")
    result.surface_chart_witness.require_current(
        result.target.mesh, result.target.geometry, target_source_query
    )
    assert result.target.associations
    transfer.validate_source(result.target)
    target_witness = result.surface_chart_witness
    standard_chart = ((0, 0), (1, 0), (0, 1))
    assert all(
        target_witness.source_parameter_chart(row) == standard_chart
        for row in range(len(target_witness.source_chart_ids))
    )
    tolerance = target_certification.inputs.fidelity_tolerance
    assert tolerance is not None
    fidelity = target_certification.fidelity
    assert fidelity is not None
    assert fidelity.mesh_to_source_upper <= tolerance
    projected = profile.project_boundary(
        np.asarray(result.target.mesh.coordinates, dtype=np.float64)
    )
    assert np.all(np.asarray(projected.resolved, dtype=np.bool_))
    # Target vertices are exact source projections, not resampled source chords.
    assert np.max(np.asarray(projected.residuals, dtype=np.float64)) <= 1.0e-12


def test_native_surface_metric_route_rebinds_trimmed_sheet_source_charts(
    _trimmed_sheet: tuple[MeshingDomain, CellMeshingResult],
) -> None:
    domain, source = _trimmed_sheet
    transfer = SurfaceAssociationTransfer(domain)
    transfer.validate_source(source)
    vertex_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    order = np.argsort(vertex_ids, kind="stable")
    vertex_set = source.mesh.entity_set(0)
    metric = MeshMetricField(
        MeshingScope(
            source.mesh.mesh_id,
            source.mesh.numeric_version,
            MeshingEntityKind.MESH,
            0,
            vertex_set.entity_set_id,
            vertex_ids[order],
        ),
        _anisotropic_metric(source.mesh)[order],
        minimum_size=0.03,
        maximum_size=0.8,
        maximum_anisotropy=2.0,
    )
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MetricMeshAdaptation(metric),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_METRIC_SURFACE,
                association_transfer=transfer,
            ),
        )
    )
    assert result.route is MeshAdaptationRoute.NATIVE_METRIC_SURFACE
    assert result.transition is not None
    assert result.transition.geometry_transition is not None
    assert result.surface_chart_witness is not None
    assert isinstance(result.evidence, SurfaceMetricOutcome)
    assert result.evidence.splits + result.evidence.collapses + result.evidence.flips > 0
    target_charts = result.target.surface_source
    if target_charts is None:
        raise AssertionError("Surface metric finalization lost its refreshed chart bank.")
    target_charts.require_root(result.target.mesh, result.target.geometry)
    assert result.transition.geometry_transition.evidence.kind == "bounded_reconstruction"

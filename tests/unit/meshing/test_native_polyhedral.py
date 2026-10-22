import itertools
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._differentiation import BranchDifferentiationPolicy
from phydrax._frozendict import frozendict
from phydrax._meshcore import (
    MeshcoreStatus,
    NativeExecutionScope,
    regular_3d,
    RestrictedPowerFailure,
)
from phydrax._model._structure import (
    model_from_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    HybridDiffusionBoundary,
    HybridMimeticDiffusion,
    PolyhedralConnectivity,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._conservation_ledger import ConservationStageLedger
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.discretization._views import FieldTracePolicy
from phydrax.discretization.finite_volume._polyhedral import (
    prepare_polyhedral_finite_volume_geometry,
)
from phydrax.discretization.finite_volume._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
)
from phydrax.discretization.vem._polyhedral import (
    prepare_polyhedral_h1_virtual_element_3d,
)
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._periodic_power_preparation import (
    PeriodicPowerImageCapacityRefusal,
    PeriodicPowerPreparation,
)
from phydrax.geometry._planar_primitives import (
    intersection,
    intersection_polygon,
    signed_measure,
)
from phydrax.geometry._supermesh import CommonRefinementStatus, prepare_common_refinement
from phydrax.geometry._triangulation import (
    PeriodicPowerSourceRefusal,
    RestrictedPowerDiagram,
)
from phydrax.lifecycle._meshing_field_records import (
    MeshingFieldStateRole,
    MeshingFiniteVolumeFieldDeclaration,
    prepare_meshing_field_index_binding,
)
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    register_meshing_source_artifacts,
    require_same_source_evidence,
    validate_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    CellMeshingResult,
    MeshingScope,
    NativePolyhedralSchedule,
    PiecewiseLinearComplex,
)
from phydrax.meshing._controls import PatchControl
from phydrax.meshing._lineage import identity_lineage
from phydrax.meshing._periodic_constraint import PeriodicConstraint
from phydrax.meshing._polyhedral_adaptation import (
    adapt_polyhedral_mesh,
    PolyhedralAdaptationEvidence,
)
from phydrax.meshing._polyhedral_generation import (
    generate_polyhedral_volume,
    PolyhedralConstruction,
    PolyhedralGenerationError,
    PolyhedralGenerationReason,
    prepare_periodic_polyhedral_seam_fragments,
)
from phydrax.meshing._topology_edit import assemble_topology_edit
from phydrax.meshing.providers._native_polyhedral import construction_failure
from phydrax.meshing.qualification._native_lifecycle import prepare_plc_associations
from phydrax.meshing.qualification._polyhedral import (
    prepare_polyhedral_archive_owners,
    prepare_polyhedral_source_star_fem,
    prepare_polyhedral_source_star_field_transfer,
    prepare_polyhedral_vem_reconstruction_transfer,
    solve_polyhedral_source_star_diffusion,
)
from tests._support.coordinate_references import (
    coordinate_reference_budget,
)
from tests._support.distributed_meshing.cold_lifecycle_identities import record_array
from tests._support.forced_devices import run_on_forced_devices
from tests._support.polyhedral_meshes import (
    periodic_cube_source,
    periodic_fv_public_inputs,
    POLYHEDRAL_CUBE_LOOPS,
    unequal_cube_seam_fixture,
    voxel_plc,
)


M = phx.meshing

# Empirical absolute tolerances from the independent volume, remap, VEM, and
# periodic-FV regressions in this file at a6827a9e54306a606ea5bc46bce894caf4ec7824.
# Values are retained unchanged; zero tolerances below denote bitwise identities.
_VOLUME_ATOL = 1e-12
_REMAP_ATOL = 1e-11
_OPERATOR_ATOL = 1e-10
_QUOTIENT_ATOL = 1e-13
# This is the authored relative stopping request, not a roundoff allowance.
_VOLUME_OPTIMIZATION_TOLERANCE = 1e-6
# Authored classification allowance in the source-action mismatch regression
# at the same revision: it exceeds 2**-30 but cannot authorize changed geometry.
_SEAM_CLASSIFICATION_TOLERANCE = 1e-6
# Authored adversarial stopping allowance: an empty required site must still
# fail even when the relative volume target allows a 200-percent error.
_LOOSE_VOLUME_TARGET_TOLERANCE = 2


def _publish(
    complex_: PiecewiseLinearComplex,
    *,
    sites: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    schedule: NativePolyhedralSchedule | None = None,
    periodic_constraints: tuple[PeriodicConstraint, ...] = (),
    patch_controls: tuple[PatchControl, ...] = (),
    region_controls: tuple[M.RegionControl, ...] = (),
) -> CellMeshingResult:
    source = (
        M.NativePlcSource(complex_, "polyhedral-domain", "r1")
        if sites is None
        else M.NativePolyhedralSource(
            complex_, "polyhedral-domain", "r1", sites=sites, weights=weights
        )
    )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "polyhedral-facets",
        np.arange(complex_.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("polyhedron",))),
        scope,
        M.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            M.UniformSizeControl(scope, 10.0, strength=M.SizeControlStrength.SOFT),
        ),
        region_controls=region_controls,
        periodic_constraints=periodic_constraints,
        patch_controls=patch_controls,
    )
    options = M.NativeMeshingOptions("plc_restricted_power", polyhedral_schedule=schedule)
    return (
        M.NativeMeshingProvider(options)
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )


def test_nonconvex_material_coverage_reciprocity_and_consumers() -> None:
    domain = voxel_plc([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 0])
    result = _publish(domain)
    assert isinstance(result, M.CellMeshingResult)
    assert result.certification is not None
    assert result.certification.passed
    fv = prepare_polyhedral_finite_volume_geometry(
        result.mesh, cell_geometry=result.geometry
    )
    vem = prepare_polyhedral_h1_virtual_element_3d(
        result.mesh, cell_geometry=result.geometry
    )
    assert float(jnp.sum(fv.cell_volumes)) == pytest.approx(3.0, abs=_VOLUME_ATOL)
    assert vem.evidence.minimum_rank_margin > 0
    assert vem.evidence.maximum_reproduction_defect < _OPERATOR_ATOL
    ones = jnp.ones(vem.dof_count)
    np.testing.assert_allclose(vem.mv(ones), 0, atol=_OPERATOR_ATOL)
    x = jnp.asarray(result.mesh.coordinates)[:, 0]
    assert float(jnp.vdot(x, vem.mv(x))) == pytest.approx(3.0, abs=_OPERATOR_ATOL)
    c = result.mesh.connectivity
    if not isinstance(c, PolyhedralConnectivity):
        raise TypeError(
            "result.mesh.connectivity must be PolyhedralConnectivity; got a different connectivity kind."
        )
    signs, faces = np.asarray(c.cell_face_sign_values), np.asarray(c.cell_face_values)
    sums = np.bincount(faces, weights=signs, minlength=c.face_count)
    counts = np.bincount(faces, minlength=c.face_count)
    np.testing.assert_array_equal(sums[counts == 2], 0)
    assert {zone.name for zone in result.zones} == {"material-0", "material-1"}
    assert any(label.name == "interface" for label in result.labels)


def test_canonical_weighted_regeneration_preserves_source_strata_and_inventory() -> None:

    domain = voxel_plc([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 0])
    source = _publish(
        domain,
        sites=np.asarray(
            [[0.25, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]], dtype=np.float64
        ),
        weights=np.asarray([0.03, -0.02, 0.01], dtype=np.float64),
    )
    original_id = source.result_id
    request = M.PolyhedralMeshAdaptation(
        M.PolyhedralAdaptationOperation.REGENERATE,
        complex_=domain,
        sites=np.asarray(
            [[0.5, 0.25, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]], dtype=np.float64
        ),
        weights=np.asarray([0.02, -0.01, 0.03], dtype=np.float64),
    )
    adapted = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            request,
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
                audit_policy=M.CellMeshAuditPolicy(
                    require_complete_association=True,
                    watertight_boundary=M.CellMeshAuditDisposition.REJECT,
                ),
            ),
        )
    )
    target = adapted.target
    assert target.certification is not None and target.certification.passed
    assert {patch.name for patch in target.patches} == {
        patch.name for patch in source.patches
    }
    assert {label.name for label in target.labels} == {
        label.name for label in source.labels
    }
    assert {(zone.name, zone.material_id, zone.region_role) for zone in target.zones} == {
        (zone.name, zone.material_id, zone.region_role) for zone in source.zones
    }
    for dimension in range(4):
        previous = next(
            value
            for value in source.associations
            if value.target_entity_set_id
            == source.mesh.entity_set(dimension).entity_set_id
        )
        current = next(
            value
            for value in target.associations
            if value.target_entity_set_id
            == target.mesh.entity_set(dimension).entity_set_id
        )
        # New cut vertices may lie on different source edges after a site move;
        # the fixed source points and higher-dimensional strata persist.
        previous_strata = {
            identity
            for identity, source_dimension in zip(
                previous.source_entities.entity_ids(),
                np.asarray(previous.source_dimensions),
                strict=True,
            )
            if dimension != 0 or source_dimension == 0
        }
        current_strata = {
            identity
            for identity, source_dimension in zip(
                current.source_entities.entity_ids(),
                np.asarray(current.source_dimensions),
                strict=True,
            )
            if dimension != 0 or source_dimension == 0
        }
        assert current_strata == previous_strata
        assert current.source_id == previous.source_id
        assert current.source_revision == previous.source_revision
    common = adapted.common_refinement
    assert common is not None and common.status is CommonRefinementStatus.SUCCESS
    old = UnstructuredFiniteVolumePlan.from_cell_mesh(source.mesh).prepare(
        cell_geometry=source.geometry
    )
    new = UnstructuredFiniteVolumePlan.from_cell_mesh(target.mesh).prepare(
        cell_geometry=target.geometry
    )
    remap = UnstructuredConservativeRemapPlan(
        old,
        new,
        common.target_offsets,
        common.source_cells,
        common.volumes,
        method="common-refinement",
        provenance=common.refinement_id,
        intersection_error_bounds=common.volume_error_bounds,
    )
    assert source.certification is not None
    values = np.where(
        np.asarray(source.certification.request.cell_regions) == 0, 2.0, 5.0
    )
    carried = np.asarray(remap.apply(values))
    for region in (0, 1):
        before = np.asarray(source.certification.request.cell_regions) == region
        after = np.asarray(target.certification.request.cell_regions) == region
        assert np.dot(
            carried[after], np.asarray(new.cell_volumes)[after]
        ) == pytest.approx(
            np.dot(values[before], np.asarray(old.cell_volumes)[before]), abs=_REMAP_ATOL
        )
    vem = prepare_polyhedral_h1_virtual_element_3d(
        target.mesh, cell_geometry=target.geometry
    )
    assert vem.evidence.minimum_rank_margin > 0
    assert vem.evidence.maximum_reproduction_defect < _OPERATOR_ATOL
    assert source.result_id == original_id


def test_regenerated_partial_facet_label_tracks_physical_overlap() -> None:

    domain = voxel_plc([(0, 0, 0)])
    published = _publish(domain)
    mesh = published.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "test.source.mesh.connectivity must be PolyhedralConnectivity; "
            f"got {type(connectivity).__name__}"
        )
    offsets = np.asarray(connectivity.face_vertex_offsets)
    vertices = np.asarray(connectivity.face_vertex_values)
    coordinates = np.asarray(mesh.coordinates)
    selected = [
        face
        for face, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
        if np.all(coordinates[vertices[a:b], 0] == 0)
    ]
    scope = M.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        M.MeshingEntityKind.MESH,
        2,
        mesh.entity_set(2).entity_set_id,
        np.asarray(connectivity.face_global_ids)[selected],
        local_mesh=mesh,
    )
    audit_policy = M.CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=M.CellMeshAuditDisposition.REJECT,
    )
    assert published.certification is not None
    source = M.certify_cell_mesh(
        mesh,
        published.coordinate_contract,
        geometry=published.geometry,
        audit_policy=audit_policy,
        patches=published.patches,
        zones=published.zones,
        labels=(*published.labels, M.MeshLabel("inlet", scope)),
        associations=published.associations,
        certification_inputs=published.certification.request,
        lineage=identity_lineage(mesh, mesh),
    )
    request = M.PolyhedralMeshAdaptation(
        M.PolyhedralAdaptationOperation.REGENERATE,
        complex_=domain,
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.asarray([0.125, -0.125], dtype=np.float64),
    )
    adapted = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            request,
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_POLYHEDRAL, audit_policy=audit_policy
            ),
        )
    )
    target = adapted.target
    assert target.certification is not None and target.certification.passed
    fv = prepare_polyhedral_finite_volume_geometry(target.mesh)
    np.testing.assert_allclose(np.sort(fv.cell_volumes), [0.25, 0.75], atol=_VOLUME_ATOL)
    inlet = next(label for label in target.labels if label.name == "inlet")
    c = target.mesh.connectivity
    if not isinstance(c, PolyhedralConnectivity):
        raise TypeError(
            "test.target.mesh.connectivity must be PolyhedralConnectivity; "
            f"got {type(c).__name__}"
        )
    rows = np.flatnonzero(
        np.isin(np.asarray(c.face_global_ids), np.asarray(inlet.scope.entity_ids))
    )
    target_points = np.asarray(target.mesh.coordinates)
    for face in rows:
        a, b = np.asarray(c.face_vertex_offsets)[face : face + 2]
        assert np.all(target_points[np.asarray(c.face_vertex_values)[a:b], 0] == 0)
    evidence = adapted.evidence
    assert isinstance(evidence, PolyhedralAdaptationEvidence)
    np.testing.assert_array_equal(evidence.site_weights, request.weights)
    np.testing.assert_array_equal(evidence.site_points, request.sites)


def test_coincident_carrier_bisector_and_conservative_nonnested_remap() -> None:
    domain = voxel_plc([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 0])
    source = _publish(domain)
    target = _publish(
        domain, sites=np.asarray([[0.5, 0.25, 0.5], [0.5, 1.75, 0.5]], dtype=np.float64)
    )
    assert source.certification is not None and target.certification is not None
    assert source.certification.passed and target.certification.passed
    refinement = prepare_common_refinement(
        source.mesh,
        target.mesh,
        source_geometry=source.geometry,
        target_geometry=target.geometry,
    )
    assert refinement.status is CommonRefinementStatus.SUCCESS
    source_connectivity = source.mesh.connectivity
    target_connectivity = target.mesh.connectivity
    if not isinstance(source_connectivity, PolyhedralConnectivity):
        raise TypeError(
            "source.mesh.connectivity must be PolyhedralConnectivity; got a different connectivity kind."
        )
    if not isinstance(target_connectivity, PolyhedralConnectivity):
        raise TypeError(
            "target.mesh.connectivity must be PolyhedralConnectivity; got a different connectivity kind."
        )
    assert source_connectivity.cell_count == 2
    assert target_connectivity.cell_count == 3
    inventory = np.array([2.0, 5.0])
    overlaps = np.asarray(refinement.volumes)
    masses = np.bincount(
        np.asarray(refinement.target_cells),
        weights=overlaps * inventory[np.asarray(refinement.source_cells)],
        minlength=target_connectivity.cell_count,
    )
    target_average = masses / np.asarray(refinement.target_measures)
    assert np.dot(
        target_average, np.asarray(refinement.target_measures)
    ) == pytest.approx(
        np.dot(inventory, np.asarray(refinement.source_measures)), abs=_VOLUME_ATOL
    )
    assert np.all((target_average >= 2) & (target_average <= 5))


def test_disconnected_components_and_hole_decomposition() -> None:
    generation_schedule = NativePolyhedralSchedule()
    with NativeExecutionScope(generation_schedule.native_limits()) as scope:
        disconnected = generate_polyhedral_volume(
            voxel_plc([(0, 0, 0), (3, 0, 0)]), schedule=generation_schedule, scope=scope
        )
    connectivity = disconnected.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "disconnected.mesh.connectivity must be PolyhedralConnectivity; got a different connectivity kind."
        )
    assert connectivity.cell_count == 2
    np.testing.assert_array_equal(disconnected.cell_sites, [0, 0])
    np.testing.assert_array_equal(disconnected.component_ids, [0, 1])
    ring = [
        (x, y, 0) for x, y in itertools.product(range(3), repeat=2) if (x, y) != (1, 1)
    ]
    result = _publish(voxel_plc(ring))
    assert result.certification is not None
    assert result.certification.passed
    fv = prepare_polyhedral_finite_volume_geometry(result.mesh)
    assert float(jnp.sum(fv.cell_volumes)) == pytest.approx(8.0, abs=_VOLUME_ATOL)
    assert np.max(np.asarray(result.mesh.coordinates)[:, 0]) == 3
    # Every star quadrature point remains outside the hole (no convex-hull fill).
    q = np.asarray(fv.cell_quadrature_points)[np.asarray(fv.cell_quadrature_valid)]
    assert not np.any((q[:, 0] > 1) & (q[:, 0] < 2) & (q[:, 1] > 1) & (q[:, 1] < 2))


def test_hidden_duplicate_sites_and_required_survival() -> None:
    domain = voxel_plc([(0, 0, 0)])
    generation_schedule = NativePolyhedralSchedule()
    with NativeExecutionScope(generation_schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            domain,
            sites=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            weights=[0, -100],
            schedule=generation_schedule,
            scope=scope,
        )
    np.testing.assert_array_equal(construction.cell_sites, [0])
    np.testing.assert_allclose(
        construction.achieved_site_volumes, [1, 0], atol=_VOLUME_ATOL
    )
    with pytest.raises(PolyhedralGenerationError, match="required_empty_sites"):
        generation_schedule = NativePolyhedralSchedule(require_every_site=True)
        with NativeExecutionScope(generation_schedule.native_limits()) as scope:
            generate_polyhedral_volume(
                domain,
                sites=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                weights=[0, -100],
                schedule=generation_schedule,
                scope=scope,
            )


def test_weight_optimization_meets_requested_site_volumes() -> None:
    generation_schedule = NativePolyhedralSchedule(target_site_volumes=(0.25, 0.75))
    with NativeExecutionScope(generation_schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            voxel_plc([(0, 0, 0)]),
            sites=[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
            schedule=generation_schedule,
            scope=scope,
        )
    np.testing.assert_allclose(
        construction.achieved_site_volumes,
        [0.25, 0.75],
        atol=_VOLUME_OPTIMIZATION_TOLERANCE,
    )
    assert construction.optimization_relative_residual <= _VOLUME_OPTIMIZATION_TOLERANCE


def test_six_hundred_collinear_sites_cover_without_all_pairs_limit() -> None:
    points = np.asarray(
        [[(i >> axis) & 1 for axis in range(3)] for i in range(8)], dtype=np.float64
    )
    tets = []
    for permutation in itertools.permutations(range(3)):
        corner = np.zeros(3, dtype=np.int64)
        tet = [0]
        for axis in permutation:
            corner[axis] = 1
            tet.append(int(corner @ np.array([1, 2, 4])))
        if np.linalg.det(points[tet[1:]] - points[tet[0]]) < 0:
            tet[0], tet[1] = tet[1], tet[0]
        tets.append(tet)
    facets = []
    for tet in tets:
        row = []
        for opposite in range(4):
            face = points[[tet[k] for k in range(4) if k != opposite]]
            token = -1
            for axis in range(3):
                for side in (0, 1):
                    if np.all(face[:, axis] == side):
                        token = 2 * axis + side
            row.append(token)
        facets.append(row)
    sites = np.column_stack(
        (np.linspace(0.01, 0.99, 600), np.full(600, 0.371), np.full(600, 0.419))
    )
    diagram = RestrictedPowerDiagram(
        sites, np.zeros(600), points, tets, np.zeros(6, dtype=np.int64), facets
    )
    data = diagram.construction
    assert np.unique(data.cell_sites).size == 600
    assert np.sum(data.cell_volumes) == pytest.approx(1.0, abs=_OPERATOR_ATOL)
    np.testing.assert_allclose(
        np.sum(data.cell_moments, axis=0), [0.5, 0.5, 0.5], atol=_OPERATOR_ATOL
    )


def test_weighted_collinear_sites_retain_their_restricted_partition() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    diagram = RestrictedPowerDiagram(
        [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
        [0.03, -0.02],
        points,
        [[0, 1, 2, 3]],
        [0],
        [[0, 1, 2, 3]],
    )
    data = diagram.construction
    volumes = np.bincount(data.cell_sites, weights=data.cell_volumes, minlength=2)
    # The weighted bisector is x=0.55; the right tetrahedron has edge scale 0.45.
    np.testing.assert_allclose(
        volumes, [1 / 6 - 0.45**3 / 6, 0.45**3 / 6], atol=_VOLUME_ATOL
    )
    assert np.sum(volumes) == pytest.approx(1 / 6, abs=_VOLUME_ATOL)


def test_weighted_cube_collinear_faces_admit_fv_and_vem() -> None:
    result = _publish(
        voxel_plc([(0, 0, 0)]),
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.asarray([0.03, -0.02], dtype=np.float64),
    )
    _assert_weighted_cube_consumer_contract(result)


def _assert_weighted_cube_consumer_contract(result: CellMeshingResult) -> None:
    fv = prepare_polyhedral_finite_volume_geometry(
        result.mesh, cell_geometry=result.geometry
    )
    vem = prepare_polyhedral_h1_virtual_element_3d(
        result.mesh, cell_geometry=result.geometry
    )
    np.testing.assert_allclose(
        np.sort(fv.cell_volumes),
        [0.45, 0.55],
        atol=_VOLUME_ATOL,
        err_msg="PolyhedralPublication.weighted_volumes",
    )
    x = np.asarray(result.mesh.coordinates)[:, 0]
    assert float(jnp.vdot(x, vem.mv(x))) == pytest.approx(1.0, abs=_OPERATOR_ATOL)
    assert vem.evidence.minimum_rank_margin > 0
    left = jnp.linspace(0.1, 1.0, vem.dof_count)
    right = jnp.linspace(1.0, -0.2, vem.dof_count)
    assert float(jnp.vdot(left, vem.mv(right))) == pytest.approx(
        float(jnp.vdot(vem.transpose_mv(left), right)), abs=_OPERATOR_ATOL
    )
    quadrature_moment = np.sum(
        np.asarray(fv.cell_quadrature_points)
        * np.asarray(fv.cell_quadrature_weights)[..., None],
        axis=(0, 1),
    )
    np.testing.assert_allclose(quadrature_moment, [0.5, 0.5, 0.5], atol=_VOLUME_ATOL)


def test_weighted_cube_oracle_detects_publication_with_wrong_weights() -> None:
    malformed = _publish(
        voxel_plc([(0, 0, 0)]),
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.zeros(2, dtype=np.float64),
    )
    with pytest.raises(AssertionError, match=r"PolyhedralPublication\.weighted_volumes"):
        _assert_weighted_cube_consumer_contract(malformed)


def test_nonconvex_face_triangulation_preserves_polynomial_geometry() -> None:
    base = np.array([[0, 0], [2, 0], [2, 2], [1, 1.5], [0, 2]], dtype=np.float64)
    n = base.shape[0]
    points = np.vstack(
        (np.column_stack((base, np.zeros(n))), np.column_stack((base, np.ones(n))))
    )
    faces = [
        np.arange(n)[::-1],
        np.arange(n, 2 * n),
        *[np.array([i, (i + 1) % n, (i + 1) % n + n, i + n]) for i in range(n)],
    ]
    mesh = phx.discretization.CellMesh.from_polyhedra(points, [faces])
    fv = prepare_polyhedral_finite_volume_geometry(mesh)
    vem = prepare_polyhedral_h1_virtual_element_3d(mesh)
    assert float(fv.cell_volumes[0]) == pytest.approx(3.5, abs=_VOLUME_ATOL)
    assert float(vem.evidence.cell_volumes[0]) == pytest.approx(3.5, abs=_VOLUME_ATOL)
    x = jnp.asarray(mesh.coordinates)[:, 0]
    assert float(jnp.vdot(x, vem.mv(x))) == pytest.approx(3.5, abs=_OPERATOR_ATOL)
    q = np.asarray(fv.face_quadrature_points)
    valid = np.asarray(fv.face_quadrature_valid)
    face_centers = np.asarray(fv.face_centers)
    # The top/bottom notch is excluded by constrained-domain triangulation.
    cap = np.abs(np.asarray(fv.face_area_vectors)[:, 2]) > 0
    cap_points = q[cap][valid[cap]]
    assert not np.any((cap_points[:, 1] > 1.5 + 0.5 * np.abs(cap_points[:, 0] - 1)))
    np.testing.assert_allclose(face_centers[cap, 0], 1, atol=_VOLUME_ATOL)


def test_exact_weighted_plane_restriction_preserves_source_and_consumer_volume() -> None:

    source = _publish(
        voxel_plc([(0, 0, 0)]),
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.asarray([0.03, -0.02], dtype=np.float64),
    )
    plane = np.asarray([1.0, 0.3, 0.2, 0.8], dtype=np.float64)
    outcome = adapt_polyhedral_mesh(
        source.mesh,
        M.PolyhedralMeshAdaptation(
            M.PolyhedralAdaptationOperation.PLANE_SPLIT, planes=plane[None, :]
        ),
        source_geometry=source.geometry,
    )
    with coordinate_reference_budget() as budget:
        mesh, _, _ = assemble_topology_edit(
            source.mesh, outcome.edit, numeric_version="exact-plane-test", budget=budget
        )
    geometry = outcome.geometry
    assert outcome.common_refinement.status is CommonRefinementStatus.SUCCESS
    assert certify_cell_geometry_validity(geometry, mesh=mesh).all_certified
    restricted = geometry.exact_source
    if not isinstance(
        restricted, phx.discretization.ExactPowerCellGeometryRestrictionSource
    ):
        raise AssertionError(
            "test.restriction.exact_source must be ExactPowerCellGeometryRestrictionSource; "
            f"got {type(restricted).__name__}"
        )
    with coordinate_reference_budget(
        queries_upper=restricted.preparation_query_upper()
    ) as budget:
        coordinates = restricted.prepare(mesh.coordinates, budget=budget).vertices
    normal = tuple(Fraction(float(value)) for value in plane[:3])
    offset = Fraction(float(plane[3]))
    for vertex in np.flatnonzero(np.asarray(restricted.vertex_plane_ids) >= 0):
        assert (
            sum(
                (
                    value * coefficient
                    for value, coefficient in zip(
                        coordinates[vertex], normal, strict=True
                    )
                ),
                Fraction(0),
            )
            == offset
        )
    fv = prepare_polyhedral_finite_volume_geometry(mesh, cell_geometry=geometry)
    vem = prepare_polyhedral_h1_virtual_element_3d(mesh, cell_geometry=geometry)
    assert float(jnp.sum(fv.cell_volumes)) == pytest.approx(1.0, abs=_VOLUME_ATOL)
    assert float(jnp.sum(vem.evidence.cell_volumes)) == pytest.approx(
        1.0, abs=_VOLUME_ATOL
    )
    assert vem.evidence.maximum_reproduction_defect < _OPERATOR_ATOL


def test_exact_power_geometry_persistence_roundtrip_keeps_power_source() -> None:

    register_meshing_source_artifacts()
    source = _publish(
        voxel_plc([(0, 0, 0)]),
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.asarray([0.03, -0.02], dtype=np.float64),
    )
    geometry = source.geometry
    if not isinstance(
        geometry.exact_source, phx.discretization.ExactPowerCellGeometrySource
    ):
        raise AssertionError(
            "test.publication.geometry.exact_source must be ExactPowerCellGeometrySource; "
            f"got {type(geometry.exact_source).__name__}"
        )
    recipe = model_structure_recipe(geometry)
    restored = model_from_array_recipe(
        recipe,
        model_recipe_array_values(geometry, recipe, prefix="power"),
        prefix="power",
    )
    validate_meshing_source_closure(restored)
    if not isinstance(
        restored.exact_source, phx.discretization.ExactPowerCellGeometrySource
    ):
        raise AssertionError(
            "test.restored.geometry.exact_source must be ExactPowerCellGeometrySource; "
            f"got {type(restored.exact_source).__name__}"
        )
    assert restored.exact_source.source_id == geometry.exact_source.source_id
    assert cell_geometry_id(restored) == cell_geometry_id(geometry)
    with coordinate_reference_budget(
        queries_upper=(
            cell_geometry_preparation_query_upper(restored)
            + cell_geometry_preparation_query_upper(geometry)
        ),
    ) as budget:
        assert restored.source_coordinates(budget=budget) == geometry.source_coordinates(
            budget=budget
        )
    fv = prepare_polyhedral_finite_volume_geometry(source.mesh, cell_geometry=restored)
    reference = prepare_polyhedral_finite_volume_geometry(
        source.mesh, cell_geometry=geometry
    )
    np.testing.assert_array_equal(
        np.asarray(fv.cell_volumes), np.asarray(reference.cell_volumes)
    )


class _PowerSourceFixture(NamedTuple):
    sites: np.ndarray
    weights: np.ndarray
    carrier: np.ndarray
    tets: np.ndarray
    offsets: np.ndarray
    equal: np.ndarray
    witnesses: np.ndarray
    source: phx.discretization.ExactPowerCellGeometrySource
    rounded: np.ndarray


@pytest.fixture(scope="module")
def power_source_fixture() -> _PowerSourceFixture:
    owner = phx.discretization.ExactPowerCellGeometrySource
    sites = np.asarray([[0.0, 0.0, 0.0], [0.7, 0.6, 0.8]], dtype=np.float64)
    weights = np.asarray([0.03, -0.02], dtype=np.float64)
    carrier = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    offsets = np.asarray([0, 1, 2, 3, 4, 6, 8, 10], dtype=np.int64)
    equal = np.asarray([0, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    witnesses = np.asarray(
        [
            [0, -1, -1, -1],
            [1, -1, -1, -1],
            [2, -1, -1, -1],
            [3, -1, -1, -1],
            [0, 3, -1, -1],
            [1, 3, -1, -1],
            [2, 3, -1, -1],
        ],
        dtype=np.int32,
    )
    source = owner(sites, weights, carrier, tets, offsets, equal, witnesses)
    with coordinate_reference_budget(
        queries_upper=source.preparation_query_upper()
    ) as budget:
        rounded = source.prepare(budget=budget).rounded_vertices
    return _PowerSourceFixture(
        sites, weights, carrier, tets, offsets, equal, witnesses, source, rounded
    )


def test_power_source_refuses_non_rne_carrier(
    power_source_fixture: _PowerSourceFixture,
) -> None:
    sites, weights, carrier, tets, offsets, equal, witnesses, source, rounded = (
        power_source_fixture
    )
    wrong_rne = rounded.copy()
    wrong_rne[4, 2] = np.nextafter(wrong_rne[4, 2], np.inf)
    with pytest.raises(
        ValueError, match=r"ExactPowerCellGeometrySource\.rounded_vertices"
    ):
        with coordinate_reference_budget(
            queries_upper=source.preparation_query_upper()
        ) as budget:
            source.prepare(wrong_rne, budget=budget)


def test_power_source_refuses_wrong_site_ancestry(
    power_source_fixture: _PowerSourceFixture,
) -> None:
    sites, weights, carrier, tets, offsets, equal, witnesses, source, rounded = (
        power_source_fixture
    )
    owner = phx.discretization.ExactPowerCellGeometrySource
    wrong_owner = equal.copy()
    wrong_owner[3] = 0
    with pytest.raises(ValueError, match=r"ExactPowerCellGeometrySource\.vertex_sites"):
        candidate = owner(sites, weights, carrier, tets, offsets, wrong_owner, witnesses)
        with coordinate_reference_budget(
            queries_upper=candidate.preparation_query_upper()
        ) as budget:
            candidate.prepare(rounded, budget=budget)


def test_power_source_refuses_deficient_intersection_rank(
    power_source_fixture: _PowerSourceFixture,
) -> None:
    sites, weights, carrier, tets, offsets, equal, witnesses, source, rounded = (
        power_source_fixture
    )
    owner = phx.discretization.ExactPowerCellGeometrySource
    wrong_rank = witnesses.copy()
    wrong_rank[4] = [0, 1, 3, -1]
    with pytest.raises(
        ValueError, match=r"ExactPowerCellGeometrySource\.vertex_carriers"
    ):
        candidate = owner(sites, weights, carrier, tets, offsets, equal, wrong_rank)
        with coordinate_reference_budget(
            queries_upper=candidate.preparation_query_upper()
        ) as budget:
            candidate.prepare(rounded, budget=budget)


def test_power_source_refuses_deficient_carrier_rank(
    power_source_fixture: _PowerSourceFixture,
) -> None:
    sites, weights, carrier, tets, offsets, equal, witnesses, source, rounded = (
        power_source_fixture
    )
    owner = phx.discretization.ExactPowerCellGeometrySource
    flat_carrier = carrier.copy()
    flat_carrier[3] = [1.0, 1.0, 0.0]
    with pytest.raises(ValueError, match=r"ExactPowerCellGeometrySource\.carrier_tets"):
        candidate = owner(sites, weights, flat_carrier, tets, offsets, equal, witnesses)
        with coordinate_reference_budget(
            queries_upper=candidate.preparation_query_upper()
        ) as budget:
            candidate.prepare(budget=budget)


@pytest.mark.parametrize(
    "maximum_work, maximum_bits, maximum_condition, diagnostic",
    [
        (1, 16384, 1e12, r"ExactPowerCellGeometrySource\.maximum_work"),
        (10_000_000, 1, 1e12, r"ExactPowerCellGeometrySource\.maximum_bits"),
        (10_000_000, 16384, 1.0, r"ExactPowerCellGeometrySource\.maximum_condition"),
    ],
    ids=["work", "integer-bits", "condition"],
)
def test_power_source_refuses_preparation_limit(
    power_source_fixture: _PowerSourceFixture,
    maximum_work: int,
    maximum_bits: int,
    maximum_condition: float,
    diagnostic: str,
) -> None:
    sites, weights, carrier, tets, offsets, equal, witnesses, _, _ = power_source_fixture
    owner = phx.discretization.ExactPowerCellGeometrySource
    with pytest.raises(ValueError, match=diagnostic):
        candidate = owner(
            sites,
            weights,
            carrier,
            tets,
            offsets,
            equal,
            witnesses,
            maximum_work=maximum_work,
            maximum_bits=maximum_bits,
            maximum_condition=maximum_condition,
        )
        with coordinate_reference_budget(
            queries_upper=candidate.preparation_query_upper()
        ) as budget:
            candidate.prepare(budget=budget)


def _independent_oriented_source_integrals(
    construction: PolyhedralConstruction, *, budget: CoordinateEnclosureBudget
) -> tuple[list[Fraction], list[list[Fraction]]]:
    """Rational divergence oracle, independent of native moments and FV/VEM."""
    c = construction.mesh.connectivity
    assert isinstance(c, PolyhedralConnectivity)
    points = np.asarray(
        construction.geometry.source_coordinates(budget=budget), dtype=object
    )
    face_offsets = np.asarray(c.face_vertex_offsets)
    face_vertices = np.asarray(c.face_vertex_values)
    cell_offsets = np.asarray(c.cell_face_offsets)
    cell_faces = np.asarray(c.cell_face_values)
    signs = np.asarray(c.cell_face_sign_values)
    volumes, moments = [], []
    for start, stop in zip(cell_offsets[:-1], cell_offsets[1:], strict=True):
        volume = Fraction(0)
        moment = [Fraction(0) for _ in range(3)]
        closed_area = [Fraction(0) for _ in range(3)]
        for face, sign in zip(cell_faces[start:stop], signs[start:stop], strict=True):
            a, b = face_offsets[face : face + 2]
            polygon = points[face_vertices[a:b]]
            for index in range(1, len(polygon) - 1):
                p, q, r = polygon[0], polygon[index], polygon[index + 1]
                cross = (
                    q[1] * r[2] - q[2] * r[1],
                    q[2] * r[0] - q[0] * r[2],
                    q[0] * r[1] - q[1] * r[0],
                )
                tetrahedron = (
                    int(sign) * sum(p[axis] * cross[axis] for axis in range(3)) / 6
                )
                volume += tetrahedron
                u, v = q - p, r - p
                area = (
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                )
                for axis in range(3):
                    closed_area[axis] += int(sign) * area[axis] / 2
                    moment[axis] += tetrahedron * (p[axis] + q[axis] + r[axis]) / 4
        assert volume > 0
        assert closed_area == [0, 0, 0]  # Constant-vector flux closure per cell.
        volumes.append(volume)
        moments.append(moment)
    face_counts = np.bincount(cell_faces, minlength=c.face_count)
    face_signs = np.bincount(cell_faces, weights=signs, minlength=c.face_count)
    assert np.all((face_counts == 1) | (face_counts == 2))
    np.testing.assert_array_equal(face_signs[face_counts == 2], 0)
    return volumes, moments


@pytest.mark.parametrize(
    "cubes",
    [
        [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        [(0, 0, 0), (3, 0, 0)],
        [(x, y, 0) for x, y in itertools.product(range(3), repeat=2) if (x, y) != (1, 1)],
    ],
    ids=["nonconvex", "disconnected", "hole"],
)
def test_declared_domain_independent_exact_volume_moment_and_flux(
    cubes: Sequence[tuple[int, int, int]],
) -> None:
    generation_schedule = NativePolyhedralSchedule()
    with NativeExecutionScope(generation_schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            voxel_plc(cubes),
            sites=[[0.25, 0.5, 0.5], [2.75, 0.5, 0.5]],
            weights=[0.125, -0.25],
            schedule=generation_schedule,
            scope=scope,
        )
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        volumes, moments = _independent_oriented_source_integrals(
            construction, budget=budget
        )
    assert sum(volumes) == len(cubes)
    for axis in range(3):
        expected = sum(
            (Fraction(cube[axis]) + Fraction(1, 2) for cube in cubes), Fraction(0)
        )
        assert sum(moment[axis] for moment in moments) == expected
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        points = np.asarray(
            construction.geometry.source_coordinates(budget=budget), dtype=object
        )
    c = construction.mesh.connectivity
    assert isinstance(c, PolyhedralConnectivity)
    offsets, vertices = (
        np.asarray(c.cell_vertex_offsets),
        np.asarray(c.cell_vertex_values),
    )
    sites = [[Fraction(float(x)) for x in row] for row in construction.diagram.points]
    weights = [Fraction(float(x)) for x in construction.diagram.weights]
    for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        owner = int(construction.cell_sites[cell])
        for point in points[vertices[a:b]]:
            powers = [
                sum((point[axis] - site[axis]) ** 2 for axis in range(3)) - weight
                for site, weight in zip(sites, weights, strict=True)
            ]
            assert powers[owner] == min(powers)


def test_concave_declared_prism_and_thin_feature_exact_oracle() -> None:
    base = np.asarray([[0, 0], [2, 0], [2, 2], [1, 1.5], [0, 2]], dtype=np.float64)
    height = 2.0**-12
    n = len(base)
    points = np.vstack(
        (
            np.column_stack((base, np.zeros(n))),
            np.column_stack((base, np.full(n, height))),
        )
    )
    polygons = [
        np.arange(n)[::-1],
        np.arange(n, 2 * n),
        *[np.asarray([i, (i + 1) % n, (i + 1) % n + n, i + n]) for i in range(n)],
    ]
    domain = PiecewiseLinearComplex(
        points,
        polygons,
        np.zeros(len(polygons), dtype=np.int64),
        [[-1, 0]],
        ("material-0",),
    )
    generation_schedule = NativePolyhedralSchedule()
    with NativeExecutionScope(generation_schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            domain, schedule=generation_schedule, scope=scope
        )
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        volumes, moments = _independent_oriented_source_integrals(
            construction, budget=budget
        )
    assert sum(volumes) == Fraction(7, 2) * Fraction(height)
    assert sum(moment[0] for moment in moments) == sum(volumes)
    assert sum(moment[2] for moment in moments) == sum(volumes) * Fraction(height) / 2
    assert np.all(construction.feature_coverage_gap == 0)
    assert np.all(construction.feature_maximum_distance == 0)


def test_required_site_survival_is_not_overridden_by_loose_volume_target() -> None:
    with pytest.raises(PolyhedralGenerationError, match="required_empty_sites"):
        generation_schedule = NativePolyhedralSchedule(
            require_every_site=True,
            target_site_volumes=(0.5, 0.5),
            relative_volume_tolerance=_LOOSE_VOLUME_TARGET_TOLERANCE,
        )
        with NativeExecutionScope(generation_schedule.native_limits()) as scope:
            generate_polyhedral_volume(
                voxel_plc([(0, 0, 0)]),
                sites=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 2.0**-10]],
                weights=[0, -100],
                schedule=generation_schedule,
                scope=scope,
            )


def test_unmet_cell_quality_is_a_bounded_failure() -> None:
    with pytest.raises(
        PolyhedralGenerationError, match="refinement_unresolved"
    ) as caught:
        generation_schedule = NativePolyhedralSchedule(
            maximum_cell_diameter=0.01,
            maximum_refinement_steps=0,
        )
        with NativeExecutionScope(generation_schedule.native_limits()) as scope:
            generate_polyhedral_volume(
                voxel_plc([(0, 0, 0)]), schedule=generation_schedule, scope=scope
            )
    assert caught.value.entities
    assert isinstance(caught.value.achieved, dict)
    assert caught.value.achieved["diameter_upper"] > 0.01


def test_native_clipping_refusal_preserves_status_failure_and_counters() -> None:

    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    sites = np.asarray(
        [[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.1, 0.6, 0.1], [0.1, 0.1, 0.6]],
        dtype=np.float64,
    )
    weights = np.zeros(4, dtype=np.float64)
    prepared, _ = regular_3d(sites, weights, max_tetrahedra=64)
    accepted = RestrictedPowerDiagram(
        sites,
        weights,
        points,
        [[0, 1, 2, 3]],
        [0],
        [[0, 1, 2, 3]],
        max_pieces=64,
    )
    clipping_capacity = len(accepted.construction.piece_sites) - 1
    assert len(prepared) <= clipping_capacity
    with pytest.raises(RestrictedPowerFailure, match="restricted_power_cells") as caught:
        RestrictedPowerDiagram(
            sites,
            weights,
            points,
            [[0, 1, 2, 3]],
            [0],
            [[0, 1, 2, 3]],
            max_pieces=clipping_capacity,
        )
    native = caught.value
    assert native.status == MeshcoreStatus.CAPACITY_EXCEEDED
    failure = construction_failure(native, "volume_fill")
    assert failure.category == M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    evidence = dict(failure.achieved)
    assert evidence["native_status"] == native.status
    for index, value in enumerate(native.failure):
        assert evidence[f"power_failure_{index}"] == value
    for index, (_, value) in enumerate(native.counters.items()):
        assert evidence[f"power_counter_{index}"] == value


def test_exact_publication_budget_refusal_preserves_actual_allowance() -> None:

    with coordinate_reference_budget(
        MeshCertificateLimits(maximum_work_units=1, maximum_scratch_bytes=1024)
    ) as ledger:
        with pytest.raises(
            CoordinateEnclosureResourceError,
            match=r"CoordinateEnclosureBudget\.coefficient_work",
        ) as caught:
            ledger.reserve(2, 0)
    failure = construction_failure(caught.value, "volume_fill")
    assert failure.category == M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.requested)["coefficient_work"] == 1
    assert dict(failure.achieved) == {"requested": 2, "completed": 0}


def test_thin_material_partition_has_exact_interface_and_region_measures() -> None:
    source = voxel_plc([(0, 0, 0), (1, 0, 0)], [0, 1])
    height = 2.0**-10
    points = np.array(source.vertices, copy=True)
    points[:, 2] *= height
    polygons = [
        source.polygon_vertices[a:b]
        for a, b in zip(
            source.polygon_offsets[:-1], source.polygon_offsets[1:], strict=True
        )
    ]
    domain = PiecewiseLinearComplex(
        points,
        polygons,
        source.polygon_facets,
        source.facet_regions,
        source.region_ids,
    )
    generation_schedule = NativePolyhedralSchedule()
    with NativeExecutionScope(generation_schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            domain,
            sites=[[0.25, 0.5, height / 2], [1.75, 0.5, height / 2]],
            weights=[0.125, 0],
            schedule=generation_schedule,
            scope=scope,
        )
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        volumes, moments = _independent_oriented_source_integrals(
            construction, budget=budget
        )
    for region in (0, 1):
        cells = np.flatnonzero(construction.cell_regions == region)
        assert sum(volumes[cell] for cell in cells) == Fraction(height)
        assert sum(moments[cell][0] for cell in cells) == Fraction(height) * Fraction(
            2 * region + 1, 2
        )
    assert construction.interface_faces.size > 0
    c = construction.mesh.connectivity
    assert isinstance(c, PolyhedralConnectivity)
    faces = np.asarray(c.cell_face_values)
    offsets = np.asarray(c.cell_face_offsets)
    signs = np.asarray(c.cell_face_sign_values)
    for face in construction.interface_faces:
        incident = []
        orientation = []
        for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            positions = np.flatnonzero(faces[a:b] == face)
            if positions.size:
                incident.append(int(construction.cell_regions[cell]))
                orientation.append(int(signs[a + positions[0]]))
        assert sorted(incident) == [0, 1]
        assert sum(orientation) == 0
    assert np.all(construction.feature_coverage_gap == 0)


def test_periodic_unequal_seam_fragments_reconstruct_both_exact_parent_actions() -> None:
    mesh, geometry, facets, source, target, matrix = unequal_cube_seam_fixture()
    constraint = M.PeriodicConstraint(
        source,
        target,
        matrix,
        tolerance=0,
        source_entity_ids=np.asarray([4], dtype=np.int64),
        orientations=np.asarray([-1], dtype=np.int64),
    )
    with coordinate_reference_budget(
        MeshCertificateLimits(
            maximum_work_units=1_000_000, maximum_scratch_bytes=16_000_000
        ),
        queries_upper=cell_geometry_preparation_query_upper(geometry),
    ) as budget:
        fragments = prepare_periodic_polyhedral_seam_fragments(
            mesh,
            geometry,
            facets,
            [constraint],
            maximum_work=1_000_000,
            maximum_scratch_bytes=16_000_000,
            budget=budget,
        )
    assert fragments
    assert sum(fragment.projected_measure for fragment in fragments) == 1
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(geometry),
    ) as budget:
        points = np.asarray(geometry.source_coordinates(budget=budget), dtype=object)
    connectivity = mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    face_ids = np.asarray(connectivity.cell_face_values)
    cell_signs = np.asarray(connectivity.cell_face_sign_values)
    outward_sign = dict(zip(map(int, face_ids), map(int, cell_signs), strict=True))
    np.testing.assert_array_equal(constraint.orientations, [-1])
    for fragment in fragments:
        assert fragment.constraint_id == constraint.constraint_id
        # Connectivity loop ordering is independent of outward cell signs.
        assert (
            fragment.relative_orientation
            * outward_sign[fragment.source_face]
            * outward_sign[fragment.target_face]
        ) == -1
        for point, source_weights, target_weights in zip(
            fragment.vertices,
            fragment.source_coefficients,
            fragment.target_coefficients,
            strict=True,
        ):
            original = tuple(
                sum(
                    weight * points[vertex, axis]
                    for vertex, weight in zip(
                        fragment.source_triangle, source_weights, strict=True
                    )
                )
                for axis in range(3)
            )
            transformed = (original[0] + 1, original[1], original[2])
            target_point = tuple(
                sum(
                    weight * points[vertex, axis]
                    for vertex, weight in zip(
                        fragment.target_triangle, target_weights, strict=True
                    )
                )
                for axis in range(3)
            )
            assert transformed == point == target_point
            assert sum(source_weights) == sum(target_weights) == 1
    source_faces = set(np.flatnonzero(facets == 4))
    assert {fragment.source_face for fragment in fragments} == source_faces
    for face in source_faces:
        assert sum(
            fragment.projected_measure
            for fragment in fragments
            if fragment.source_face == face
        ) == Fraction(1, 2)


def test_periodic_seam_tolerance_does_not_replace_original_source_transform() -> None:
    mesh, geometry, facets, source, target, matrix = unequal_cube_seam_fixture()
    matrix[0, 3] += 2.0**-30
    # The classification allowance exceeds the authored 2**-30 action mismatch;
    # source coverage must still refuse that mismatch.
    constraint = M.PeriodicConstraint(
        source, target, matrix, tolerance=_SEAM_CLASSIFICATION_TOLERANCE
    )
    with pytest.raises(PolyhedralGenerationError, match="periodic_seam_exact_coverage"):
        with coordinate_reference_budget(
            MeshCertificateLimits(
                maximum_work_units=1_000_000, maximum_scratch_bytes=16_000_000
            ),
            queries_upper=cell_geometry_preparation_query_upper(geometry),
        ) as budget:
            prepare_periodic_polyhedral_seam_fragments(
                mesh,
                geometry,
                facets,
                [constraint],
                maximum_work=1_000_000,
                maximum_scratch_bytes=16_000_000,
                budget=budget,
            )
    assert float(np.asarray(constraint.transform)[0, 3]) == 1 + 2.0**-30


def test_periodic_seam_common_refinement_refuses_original_work_capacity() -> None:

    mesh, geometry, facets, source, target, matrix = unequal_cube_seam_fixture()
    constraint = M.PeriodicConstraint(source, target, matrix, tolerance=0)
    with pytest.raises(
        CoordinateEnclosureResourceError,
        match=r"CoordinateEnclosureBudget\.coefficient_work",
    ) as caught:
        with coordinate_reference_budget(
            MeshCertificateLimits(maximum_work_units=1, maximum_scratch_bytes=16_000_000),
            queries_upper=cell_geometry_preparation_query_upper(geometry),
        ) as budget:
            prepare_periodic_polyhedral_seam_fragments(
                mesh,
                geometry,
                facets,
                [constraint],
                maximum_work=1,
                maximum_scratch_bytes=16_000_000,
                budget=budget,
            )
    assert caught.value.limit == 1
    assert caught.value.requested > caught.value.completed


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_canonical_exact_clip_polygon_keeps_measure_and_oriented_loop(
    reverse: bool,
) -> None:

    subject = tuple(
        tuple(Fraction(x) for x in point) for point in [(0, 0), (2, 0), (2, 1), (0, 1)]
    )
    target = tuple(
        tuple(Fraction(x) for x in point) for point in [(1, -1), (3, -1), (3, 2), (1, 2)]
    )
    if reverse:
        subject = subject[::-1]
    with coordinate_reference_budget() as budget:
        polygon = intersection_polygon(subject, target, budget=budget)
        assert (
            abs(signed_measure(polygon, budget=budget))
            == intersection(subject, target, budget=budget)
            == 1
        )
        assert (
            signed_measure(polygon, budget=budget)
            * signed_measure(subject, budget=budget)
            > 0
        )
        assert (
            intersection_polygon(
                ((Fraction(0),), (Fraction(1),)),
                ((Fraction(2),), (Fraction(3),)),
                budget=budget,
            )
            == ()
        )


def test_periodic_image_capacity_refusal_keeps_original_source_bits_and_limit() -> None:

    points = np.asarray([[0.25, -0.0, 0.25]])
    weights = np.asarray([-0.0])
    carrier = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    with pytest.raises(
        PeriodicPowerImageCapacityRefusal, match=r"PeriodicPowerPreparation\."
    ) as caught:
        PeriodicPowerPreparation(
            points, weights, carrier, PeriodicCell(np.eye(3)), maximum_images=1
        )
    error = caught.value
    np.testing.assert_array_equal(error.points.view(np.uint64), points.view(np.uint64))
    np.testing.assert_array_equal(error.weights.view(np.uint64), weights.view(np.uint64))
    failure = construction_failure(error, "volume_fill")
    assert failure.category == M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.requested)["site_images"] == error.maximum_images == 1
    assert dict(failure.achieved)["requested_site_images"] == error.requested_images > 1
    assert dict(failure.achieved)["completed_site_images"] == error.completed_images


def test_periodic_adjacency_refusal_keeps_source_identity_and_actual_work_bound() -> None:

    points = np.asarray([[0.25, 0.25, 0.25]])
    weights = np.asarray([0.0])
    carrier = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    group = PeriodicCell(np.eye(3))
    preparation = PeriodicPowerPreparation(
        points,
        weights,
        carrier,
        group,
        maximum_images=4096,
        maximum_work_units=2_000_000,
    )
    with pytest.raises(
        PeriodicPowerSourceRefusal, match=r"PeriodicPowerSourceRefusal\.construction"
    ) as caught:
        RestrictedPowerDiagram(
            points,
            weights,
            carrier,
            [[0, 1, 2, 3]],
            [0],
            [[0, 1, 2, 3]],
            periodic_group=group,
            periodic_preparation=preparation,
            work_limit=1,
        )
    error = caught.value
    assert error.preparation.source_id == preparation.source_id
    failure = construction_failure(error, "volume_fill")
    assert failure.category == M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert preparation.source_id in str(failure)
    assert dict(failure.requested)["periodic_call_work_units"] == error.maximum_work == 1
    assert dict(failure.achieved)["periodic_requested_work_units"] == error.requested_work
    assert dict(failure.achieved)["periodic_completed_work_units"] == error.completed_work


@pytest.fixture(scope="module")
def periodic_power_archive(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    PeriodicPowerPreparation,
    tuple[PeriodicPowerPreparation, ...],
    dict[str, Any],
]:
    tmp_path = tmp_path_factory.mktemp("periodic-power-archive")
    points = np.asarray([[0.25, -0.0, 0.25]])
    weights = np.asarray([-0.0])
    carrier = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    preparation = PeriodicPowerPreparation(
        points,
        weights,
        carrier,
        PeriodicCell(np.eye(3)),
        maximum_images=257,
        maximum_work_units=1 << 26,
    )
    limits = ArrayArchiveLimits()
    receipt = write_meshing_source_closure(
        tmp_path / "periodic-power",
        (preparation, preparation),
        limits=limits,
    )
    cold = run_on_forced_devices(
        "tests._support.distributed_meshing.cold_lifecycle",
        "periodic_power_preparation",
        (str(tmp_path / "periodic-power"), receipt.content_id),
        devices=1,
        env={},
    )
    # Consumer assertions stay in the tests; one cold reader is shared by
    # identity and independently collected malformed-restoration cases.
    restored = read_meshing_source_closure(
        tmp_path / "periodic-power",
        expected_content_id=receipt.content_id,
        limits=limits,
    )
    return preparation, restored, cold


def test_periodic_power_preparation_cold_archive_preserves_original_law(
    periodic_power_archive: tuple[
        PeriodicPowerPreparation,
        tuple[PeriodicPowerPreparation, ...],
        dict[str, Any],
    ],
) -> None:
    _assert_periodic_power_archive_law(*periodic_power_archive)


def _assert_periodic_power_archive_law(
    preparation: PeriodicPowerPreparation,
    restored: tuple[PeriodicPowerPreparation, ...],
    cold: dict[str, Any],
) -> None:
    assert cold["type_exact"]
    assert cold["alias"]
    owner = restored[0]
    assert owner is restored[1], "PeriodicPowerArchive.alias"
    assert type(owner) is PeriodicPowerPreparation
    assert owner.preparation_id == preparation.preparation_id
    assert owner.maximum_images == preparation.maximum_images
    assert owner.maximum_work_units == preparation.maximum_work_units
    for name in ("points", "weights", "domain_points"):
        np.testing.assert_array_equal(
            np.asarray(getattr(owner, name)).view(np.uint64),
            np.asarray(getattr(preparation, name)).view(np.uint64),
        )


def test_periodic_power_archive_oracle_detects_lost_alias(
    periodic_power_archive: tuple[
        PeriodicPowerPreparation,
        tuple[PeriodicPowerPreparation, ...],
        dict[str, Any],
    ],
) -> None:
    preparation, restored, cold = periodic_power_archive
    owner = restored[0]
    independent_owner = _corrupt_record_path(
        owner, ("maximum_images",), owner.maximum_images
    )
    with pytest.raises(AssertionError, match=r"PeriodicPowerArchive\.alias"):
        _assert_periodic_power_archive_law(preparation, (owner, independent_owner), cold)


def _shift_first_periodic_image_action(owner: PeriodicPowerPreparation) -> np.ndarray:
    actions = np.array(owner.image_exponents, copy=True)
    actions[0, 0] += 1
    return actions


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("image_exponents", _shift_first_periodic_image_action),
        ("carrier_id", lambda owner: "not-the-original-carrier"),
    ],
    ids=["image-exponents", "carrier-id"],
)
def test_periodic_power_preparation_refuses_stored_corruption(
    periodic_power_archive: tuple[
        PeriodicPowerPreparation,
        tuple[PeriodicPowerPreparation, ...],
        dict[str, Any],
    ],
    field: str,
    replacement: Callable[[PeriodicPowerPreparation], object],
) -> None:
    _, restored, _ = periodic_power_archive
    owner = restored[0]
    with pytest.raises(ValueError, match=rf"PeriodicPowerPreparation\.{field}"):
        corrupted = _corrupt_record_path(owner, (field,), replacement(owner))
        validate_meshing_source_closure(corrupted, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)


def test_periodic_image_action_fault_changes_only_first_exponent(
    periodic_power_archive: tuple[
        PeriodicPowerPreparation,
        tuple[PeriodicPowerPreparation, ...],
        dict[str, Any],
    ],
) -> None:
    _, restored, _ = periodic_power_archive
    owner = restored[0]
    original = np.array(owner.image_exponents, copy=True)
    altered = _shift_first_periodic_image_action(owner)
    difference = altered - original
    assert np.count_nonzero(difference) == 1
    assert difference[0, 0] == 1
    np.testing.assert_array_equal(owner.image_exponents, original)


def _source_face_area_vector(
    construction: PolyhedralConstruction, face: int, *, budget: CoordinateEnclosureBudget
) -> tuple[Fraction, ...]:
    c = construction.mesh.connectivity
    assert isinstance(c, PolyhedralConnectivity)
    points = np.asarray(
        construction.geometry.source_coordinates(budget=budget), dtype=object
    )
    a, b = np.asarray(c.face_vertex_offsets)[face : face + 2]
    polygon = points[np.asarray(c.face_vertex_values)[a:b]]
    vector = [Fraction(0)] * 3
    for index in range(1, len(polygon) - 1):
        u, v = polygon[index] - polygon[0], polygon[index + 1] - polygon[0]
        vector[0] += (u[1] * v[2] - u[2] * v[1]) / 2
        vector[1] += (u[2] * v[0] - u[0] * v[2]) / 2
        vector[2] += (u[0] * v[1] - u[1] * v[0]) / 2
    owner = int(np.asarray(c.face_owner)[face])
    first, last = np.asarray(c.cell_face_offsets)[owner : owner + 2]
    position = np.flatnonzero(np.asarray(c.cell_face_values)[first:last] == face)
    sign = int(np.asarray(c.cell_face_sign_values)[first + position[0]])
    return tuple(sign * value for value in vector)


def test_full_periodic_power_source_independent_measure_faces_and_flux() -> None:
    domain, constraints = periodic_cube_source()
    generation_schedule = NativePolyhedralSchedule()
    with NativeExecutionScope(generation_schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            domain,
            sites=[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
            weights=[0, 0],
            periodic_constraints=constraints,
            schedule=generation_schedule,
            scope=scope,
        )
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        volumes, moments = _independent_oriented_source_integrals(
            construction, budget=budget
        )
    assert sum(volumes) == 1
    assert (
        tuple(sum(moment[axis] for moment in moments) for axis in range(3))
        == (Fraction(1, 2),) * 3
    )
    assert construction.mesh.periodic_topology is not None
    assert construction.seam_face_pairs is not None
    assert construction.seam_vertex_pairs is not None
    for source_face, target_face, generator in construction.seam_face_pairs:
        with coordinate_reference_budget(
            queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
        ) as budget:
            source = _source_face_area_vector(
                construction, int(source_face), budget=budget
            )
        with coordinate_reference_budget(
            queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
        ) as budget:
            target = _source_face_area_vector(
                construction, int(target_face), budget=budget
            )
        matrix = np.asarray(constraints[generator].transform)
        rotated = tuple(
            sum(
                Fraction(float(matrix[axis, column])) * source[column]
                for column in range(3)
            )
            for axis in range(3)
        )
        assert rotated == tuple(-value for value in target)
        velocity = (Fraction(2), Fraction(-3), Fraction(5))
        target_velocity = tuple(
            sum(
                Fraction(float(matrix[axis, column])) * velocity[column]
                for column in range(3)
            )
            for axis in range(3)
        )
        assert (
            sum(
                source[axis] * velocity[axis] + target[axis] * target_velocity[axis]
                for axis in range(3)
            )
            == 0
        )
    assert np.all(construction.feature_coverage_gap == 0)
    assert np.all(construction.feature_maximum_distance == 0)


def test_public_periodic_power_route_retains_original_source_and_patch_conformity() -> (
    None
):
    domain, constraints = periodic_cube_source()
    patch = PatchControl("periodic-left", constraints[0].source_scope, ("material-0",))
    region = M.RegionControl(
        M.MeshingScope(
            "polyhedral-domain",
            "r1",
            M.MeshingEntityKind.GEOMETRY,
            3,
            "material-regions",
            [0],
        ),
        "material-0",
        "material-0",
        M.RegionRole.SOLID,
    )
    result = _publish(
        domain,
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.zeros(2, dtype=np.float64),
        periodic_constraints=constraints,
        patch_controls=(patch,),
        region_controls=(region,),
    )
    assert result.certification is not None and result.certification.passed
    assert result.mesh.periodic_topology is not None
    assert any(value.name == "periodic" for value in result.labels)
    named = next(value for value in result.patches if value.name == patch.name)
    assert named.scope.entity_ids.size
    assert named.adjacent_zone_ids == tuple(zone.zone_id for zone in result.zones)
    source = result.geometry.exact_source
    assert isinstance(source, phx.discretization.ExactPowerCellGeometryLinearActionSource)
    assert isinstance(source.parent, phx.discretization.ExactPowerCellGeometrySource)

    assert isinstance(source.periodic_preparation, PeriodicPowerPreparation)
    assert isinstance(source.parent.periodic_preparation, PeriodicPowerPreparation)
    np.testing.assert_array_equal(
        source.parent.site_points, [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]
    )
    np.testing.assert_array_equal(source.parent.site_weights, [0, 0])
    assert (
        source.periodic_preparation.source_id
        == source.parent.periodic_preparation.source_id
    )


def test_sheared_periodic_power_publication_uses_exact_source_not_double_rne_lifts() -> (
    None
):

    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.float64
    )
    points[:, 0] += points[:, 1]
    domain = PiecewiseLinearComplex(
        points, POLYHEDRAL_CUBE_LOOPS, np.arange(6), [(-1, 0)] * 6, ("material-0",)
    )
    source_scope = MeshingScope(
        "sheared-power", "r1", M.MeshingEntityKind.GEOMETRY, 2, "source", [5]
    )
    target_scope = MeshingScope(
        "sheared-power", "r1", M.MeshingEntityKind.GEOMETRY, 2, "target", [4]
    )
    matrix = np.eye(4)
    matrix[0, 3] = -1
    constraint = PeriodicConstraint(
        source_scope,
        target_scope,
        matrix,
        tolerance=0,
        source_entity_ids=np.asarray([5], dtype=np.int64),
        orientations=np.asarray([-1], dtype=np.int64),
    )
    generation_schedule = NativePolyhedralSchedule()
    with NativeExecutionScope(generation_schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            domain,
            sites=[[0, 0, 0.5], [0, 3, 0.5]],
            weights=[0, 7],
            periodic_constraints=[constraint],
            schedule=generation_schedule,
            scope=scope,
        )
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        volumes, moments = _independent_oriented_source_integrals(
            construction, budget=budget
        )
    assert sum(volumes) == 1
    assert tuple(sum(moment[axis] for moment in moments) for axis in range(3)) == (
        Fraction(1),
        Fraction(1, 2),
        Fraction(1, 2),
    )
    original = construction.mesh
    topology = original.periodic_topology
    assert topology is not None
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        exact = construction.geometry.source_coordinates(budget=budget)
    chosen = next(
        index
        for index, point in enumerate(exact)
        if point[0] == Fraction(4, 3) and point[1] == Fraction(1, 3)
    )
    representatives = np.array(topology.vertex_representatives, copy=True)
    shifts = np.array(topology.vertex_shifts, copy=True)
    root = representatives[chosen]
    members = np.flatnonzero(representatives == root)
    origin = shifts[chosen].copy()
    representatives[members] = chosen
    shifts[members] -= origin
    assert isinstance(original.connectivity, PolyhedralConnectivity)
    assert isinstance(
        construction.geometry.exact_source,
        phx.discretization.ExactPowerCellGeometryLinearActionSource,
    )
    lifted = CellMesh(
        original.coordinates,
        original.blocks,
        vertex_global_ids=original.vertex_global_ids,
        polyhedral_connectivity=original.connectivity,
        numeric_version=original.numeric_version,
    )
    geometry = CellGeometrySpec.power(lifted, construction.geometry.exact_source)
    with pytest.raises(ValueError, match="coordinates"):
        PeriodicMeshTopology(lifted, topology.cell, representatives, shifts)
    renewed = PeriodicMeshTopology(
        lifted, topology.cell, representatives, shifts, actual_geometry=geometry
    )
    assert renewed.actual_geometry is not None
    for member in members:
        assert renewed.vertex_representatives[member] == chosen


def _collect_shared_recipe_nodes(node: Any, shared: list[dict[str, Any]]) -> None:
    """Visit only DAG-valued children in the public recipe's ordinal order."""
    if isinstance(node, dict):
        kind = node.get("kind")
        # Codec payloads, set items and mapping keys have no shared ordinals.
        if kind in ("value", "set", "frozenset"):
            return
        if kind in ("mapping", "frozendict", "mappingproxy"):
            for _, child in node["items"]:
                _collect_shared_recipe_nodes(child, shared)
        else:
            for child in node.values():
                _collect_shared_recipe_nodes(child, shared)
        if kind in ("array", "prng_key", "dataclass"):
            shared.append(node)
    elif isinstance(node, list):
        for child in node:
            _collect_shared_recipe_nodes(child, shared)


def _corrupt_record_path[T](
    value: T, path: tuple[str | int, ...], replacement: object
) -> T:
    """Restore malformed public recipe data, without mutating frozen owners."""
    register_meshing_source_artifacts()
    parent: Any = value
    for component in path[:-1]:
        parent = (
            parent[component]
            if isinstance(component, int)
            else getattr(parent, component)
        )
    if not is_dataclass(parent) or isinstance(parent, type):
        raise TypeError(
            "corruption.parent must be a dataclass instance; got an unsupported kind"
        )
    name = path[-1]
    if not isinstance(name, str):
        raise TypeError(
            "corruption.field must name a declared dataclass field; got a non-string selector"
        )
    declared = tuple(field.name for field in fields(parent))
    if name not in declared:
        raise ValueError(f"corruption.field must be declared; got {name!r}")
    position = declared.index(name)
    original = getattr(parent, name)
    # The second tuple item references the already encoded parent. Its ordinal
    # identifies the first declaration even when the selected path is an alias.
    recipe = model_structure_recipe((value, parent))
    arrays = model_recipe_array_values((value, parent), recipe, prefix="corruption")
    shared: list[dict[str, Any]] = []

    _collect_shared_recipe_nodes(recipe, shared)
    target = shared[recipe["items"][1]["target"]]
    if isinstance(original, (Array, np.ndarray)):
        name = next(name for name, payload in arrays.items() if payload is original)
        arrays[name] = np.asarray(replacement)
    else:
        # The selected scalar keeps the shared-object ordinals and array
        # inventory intact; changing compound structure is not this fault's job.
        if not isinstance(original, (str, int, float, bool)) or not isinstance(
            replacement, (str, int, float, bool)
        ):
            raise TypeError(
                "corruption.replacement must preserve scalar or array structure; got incompatible values"
            )
        target["items"][position] = model_structure_recipe(replacement)
    return model_from_array_recipe(recipe, arrays, prefix="corruption")[0]


class _PeriodicFVScenario(NamedTuple):
    action_kind: str
    generation_source: M.NativePolyhedralSource
    specification: M.VolumeMeshingSpec
    options: M.NativeMeshingOptions
    action: np.ndarray
    construction: PolyhedralConstruction
    original_coordinates: np.ndarray
    original_source: tuple[tuple[Fraction, ...], ...]
    system: phx.equations.ScalarConservationSystem
    prepared: phx.discretization.UnstructuredFiniteVolumeDiscretization
    diffusion: phx.discretization.HybridMimeticDiffusion
    method: phx.discretization.UnstructuredFiniteVolumeMethodPlan
    boundaries: phx.discretization.UnstructuredFiniteVolumeBoundarySet
    dynamics: phx.discretization.PreparedUnstructuredFiniteVolumeDynamics


class _ScalarFVEvolution(NamedTuple):
    state: Array
    history: tuple[Array, ...]
    residuals: tuple[Array, ...]
    varying: Array
    manufactured_history: tuple[Array, ...]
    initial_mode: Array | None
    trace_histories: tuple[Array, ...]


class _VectorFVEvolution(NamedTuple):
    system: phx.equations.EulerSystem
    prepared: phx.discretization.UnstructuredFiniteVolumeDiscretization
    dynamics: phx.discretization.PreparedUnstructuredFiniteVolumeDynamics
    initial_state: Array
    state: Array
    history: tuple[Array, ...]


@pytest.fixture(scope="module", params=["translation", "rotation", "screw"])
def periodic_fv_case(request: pytest.FixtureRequest) -> _PeriodicFVScenario:
    action_kind = request.param
    generation_source, specification, options, action = periodic_fv_public_inputs(
        action_kind
    )
    domain = generation_source.complex
    constraints = specification.periodic_constraints
    generation_schedule = NativePolyhedralSchedule()
    with NativeExecutionScope(generation_schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            domain,
            sites=[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
            weights=[0.0, 0.0],
            periodic_constraints=constraints,
            source_id="polyhedral-domain",
            schedule=generation_schedule,
            scope=scope,
        )
    original_coordinates = np.asarray(construction.mesh.coordinates).copy()
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        original_source = construction.geometry.rational_source_coordinates(budget=budget)
    velocity = (0.0, 0.0, 1.0) if action_kind == "rotation" else (1.0, 0.0, 0.0)
    system = phx.equations.ScalarConservationSystem(
        3,
        lambda value, axis, args: velocity[axis] * value,
        lambda left, right, axis, args: jnp.full(left.shape[:-1], abs(velocity[axis])),
        system_id="source-periodic-polyhedral-fv-advection",
    )
    prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(
        construction.mesh,
        component_names=system.component_names,
    ).prepare(
        cell_geometry=construction.geometry,
    )
    diffusion = HybridMimeticDiffusion(prepared)
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        prepared.boundary_patch_names,
        {"boundary": phx.discretization.ExtrapolationBoundary()},
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    dynamics = phx.discretization.PreparedUnstructuredFiniteVolumeDynamics(
        system,
        prepared,
        method,
        boundaries,
    )
    return _PeriodicFVScenario(
        action_kind,
        generation_source,
        specification,
        options,
        action,
        construction,
        original_coordinates,
        original_source,
        system,
        prepared,
        diffusion,
        method,
        boundaries,
        dynamics,
    )


@pytest.fixture(scope="module")
def scalar_fv_evolution(periodic_fv_case: _PeriodicFVScenario) -> _ScalarFVEvolution:
    prepared, dynamics = periodic_fv_case.prepared, periodic_fv_case.dynamics
    state = jnp.ones(prepared.state_shape)
    history, residuals = [], []
    for step in range(4):
        residual = dynamics(jnp.asarray(step * 0.01), state)
        residuals.append(residual)
        state = state + 0.01 * residual
        history.append(state)
    varying = jnp.arange(prepared.cell_count, dtype=jnp.float64)[:, None] + 1
    manufactured_history = []
    initial_mode = None
    if periodic_fv_case.action_kind != "rotation":
        assert prepared.cell_count == 2
        order = np.argsort(np.asarray(prepared.cell_centers)[:, 0])
        manufactured = jnp.zeros((2, 1)).at[order].set(jnp.asarray(((1.0,), (2.0,))))
        initial_mode = manufactured - 1.5
        for step in range(4):
            manufactured = manufactured + 0.01 * dynamics(
                jnp.asarray(step * 0.01), manufactured
            )
            manufactured_history.append(manufactured)
    trace_histories = tuple(
        manufactured_history[:3] if manufactured_history else history[:3]
    )
    return _ScalarFVEvolution(
        state,
        tuple(history),
        tuple(residuals),
        varying,
        tuple(manufactured_history),
        initial_mode,
        trace_histories,
    )


@pytest.fixture(scope="module")
def vector_fv_evolution(periodic_fv_case: _PeriodicFVScenario) -> _VectorFVEvolution:
    construction, action_kind = (
        periodic_fv_case.construction,
        periodic_fv_case.action_kind,
    )
    method, boundaries = periodic_fv_case.method, periodic_fv_case.boundaries
    euler = phx.equations.EulerSystem(3)
    vector_prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(
        construction.mesh,
        component_names=euler.component_names,
    ).prepare(cell_geometry=construction.geometry)
    vector_dynamics = phx.discretization.PreparedUnstructuredFiniteVolumeDynamics(
        euler,
        vector_prepared,
        method,
        boundaries,
    )
    conserved_constant = (
        (1.0, 0.0, 0.0, 0.2, 3.0)
        if action_kind == "rotation"
        else (1.0, 0.2, 0.0, 0.0, 3.0)
    )
    vector_state = jnp.broadcast_to(
        jnp.asarray(conserved_constant), vector_prepared.state_shape
    )
    initial_state = vector_state
    vector_history = []
    for step in range(4):
        vector_state = vector_state + 0.01 * vector_dynamics(
            jnp.asarray(step * 0.01), vector_state
        )
        vector_history.append(vector_state)

    return _VectorFVEvolution(
        euler,
        vector_prepared,
        vector_dynamics,
        initial_state,
        vector_state,
        tuple(vector_history),
    )


@pytest.fixture(scope="module")
def periodic_fv_publication(periodic_fv_case: _PeriodicFVScenario) -> CellMeshingResult:
    return (
        M.NativeMeshingProvider(periodic_fv_case.options)
        .plan(
            periodic_fv_case.generation_source,
            periodic_fv_case.specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )


def test_periodic_polyhedral_fv_capacity_refusal_preserves_source(
    periodic_fv_case: _PeriodicFVScenario,
) -> None:
    construction = periodic_fv_case.construction
    original_coordinates, original_source = (
        periodic_fv_case.original_coordinates,
        periodic_fv_case.original_source,
    )
    with pytest.raises(ValueError, match=r"prepare_polyhedral_worksets\.maximum_entries"):
        prepare_polyhedral_finite_volume_geometry(
            construction.mesh,
            cell_geometry=construction.geometry,
            maximum_workset_entries=1,
        )
    np.testing.assert_array_equal(
        np.asarray(construction.mesh.coordinates).view(np.uint64),
        original_coordinates.view(np.uint64),
    )
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        assert construction.geometry.source_coordinates(budget=budget) == original_source


def test_periodic_polyhedral_fv_quotient_geometry_and_flux_adjoint(
    periodic_fv_case: _PeriodicFVScenario,
) -> None:
    construction, prepared, diffusion = (
        periodic_fv_case.construction,
        periodic_fv_case.prepared,
        periodic_fv_case.diffusion,
    )
    assert construction.seam_face_pairs is not None
    pairs = construction.seam_face_pairs[:, :2]
    active = np.asarray(prepared.face_block.active_mask)
    for left, right in pairs:
        rows = np.flatnonzero(
            np.isin(np.asarray(prepared.source_face_indices), (left, right))
        )
        assert rows.size == 1
        assert bool(active[rows[0]])
        assert int(prepared.neighbor_cells[rows[0]]) >= 0
        assert int(prepared.boundary_patch_ids[rows[0]]) == -1
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        volumes, moments = _independent_oriented_source_integrals(
            construction, budget=budget
        )
    np.testing.assert_allclose(
        prepared.cell_volumes, [float(value) for value in volumes], atol=_QUOTIENT_ATOL
    )
    np.testing.assert_allclose(
        prepared.cell_centers,
        [
            [float(value / volume) for value in moment]
            for volume, moment in zip(volumes, moments, strict=True)
        ],
        atol=_QUOTIENT_ATOL,
    )
    constant = jnp.ones((prepared.cell_count,))
    traces = jnp.ones((diffusion.face_count,))
    local = diffusion.local_fluxes(constant, traces, jnp.eye(3))
    np.testing.assert_allclose(local, 0, atol=_QUOTIENT_ATOL)
    mask = active & (np.asarray(prepared.neighbor_cells) >= 0)
    rates = jnp.asarray(np.arange(diffusion.face_count, dtype=np.float64) * mask)
    result = diffusion.cell_divergence(rates)
    np.testing.assert_allclose(jnp.sum(result), 0, atol=_QUOTIENT_ATOL)
    direction = jnp.linspace(-0.5, 0.5, diffusion.face_count)
    cotangent = jnp.arange(prepared.cell_count, dtype=jnp.float64) + 0.25
    _, tangent = jax.jvp(diffusion.cell_divergence, (rates,), (direction,))
    _, pullback = jax.vjp(diffusion.cell_divergence, rates)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, tangent),
        jnp.vdot(pullback(cotangent)[0], direction),
        atol=_QUOTIENT_ATOL,
    )


def test_periodic_polyhedral_scalar_constant_history_and_conservation(
    periodic_fv_case: _PeriodicFVScenario,
    scalar_fv_evolution: _ScalarFVEvolution,
) -> None:
    prepared, dynamics = periodic_fv_case.prepared, periodic_fv_case.dynamics
    for residual in scalar_fv_evolution.residuals:
        np.testing.assert_allclose(residual, 0, atol=_QUOTIENT_ATOL)
    np.testing.assert_allclose(
        jnp.stack(scalar_fv_evolution.history), 1, atol=_QUOTIENT_ATOL
    )
    varying = scalar_fv_evolution.varying
    residual = dynamics(jnp.asarray(0.0), varying)
    np.testing.assert_allclose(
        jnp.sum(prepared.cell_volumes[:, None] * residual), 0, atol=_QUOTIENT_ATOL
    )


@pytest.mark.parametrize("periodic_fv_case", ["translation", "screw"], indirect=True)
def test_periodic_polyhedral_scalar_manufactured_upwind_mode(
    periodic_fv_case: _PeriodicFVScenario,
    scalar_fv_evolution: _ScalarFVEvolution,
) -> None:
    prepared, dynamics = periodic_fv_case.prepared, periodic_fv_case.dynamics
    assert prepared.cell_count == 2
    initial_mode = scalar_fv_evolution.initial_mode
    assert initial_mode is not None
    np.testing.assert_allclose(
        dynamics(jnp.asarray(0.0), initial_mode + 1.5),
        -4 * initial_mode,
        atol=_VOLUME_ATOL,
    )
    for step, state in enumerate(scalar_fv_evolution.manufactured_history):
        np.testing.assert_allclose(
            state, 1.5 + initial_mode * (1 - 4 * 0.01) ** (step + 1), atol=_VOLUME_ATOL
        )


@pytest.mark.parametrize("periodic_fv_case", ["translation", "screw"], indirect=True)
def test_periodic_polyhedral_diffusion_mode_and_pinned_physical_residual(
    periodic_fv_case: _PeriodicFVScenario,
) -> None:
    prepared, diffusion = periodic_fv_case.prepared, periodic_fv_case.diffusion
    native_policy = phx.linalg.LinearSolvePolicy(phx.linalg.ConjugateGradient())
    order = np.argsort(np.asarray(prepared.cell_centers)[:, 0])
    # The same source-frame mimetic operator gives the independent
    # two-half-cell diffusion mode du/dt=-16(u-mean), with no seam law.
    diffused = jnp.zeros((2,)).at[order].set(jnp.asarray((1.0, 2.0)))
    diffusion_mode = diffused - 1.5
    for step in range(4):
        owner_values = diffused[diffusion.owner_cells]
        neighbor_values = diffused[jnp.maximum(diffusion.neighbor_cells, 0)]
        face_values = jnp.where(
            diffusion.neighbor_cells >= 0,
            (owner_values + neighbor_values) / 2,
            owner_values,
        )
        face_rates = diffusion.face_fluxes(diffused, face_values, jnp.eye(3))
        rate = -diffusion.cell_divergence(face_rates) / prepared.cell_volumes
        np.testing.assert_allclose(rate, -16 * (diffused - 1.5), atol=_VOLUME_ATOL)
        diffused = diffused + 0.01 * rate
        np.testing.assert_allclose(
            diffused,
            1.5 + diffusion_mode * (1 - 16 * 0.01) ** (step + 1),
            atol=_VOLUME_ATOL,
        )
    neumann = HybridDiffusionBoundary(prepared)
    # Authored piecewise physical forcing -u''=(-.25,+.25), integrated
    # over the two half cells, independently of the discrete A*u action.
    loads = jnp.zeros((2,)).at[order].set(jnp.asarray((-0.125, 0.125)))
    pinned_system, pinned_rhs = diffusion.pinned_linear_system(
        jnp.eye(3),
        neumann,
        source=loads,
        pin_cell=int(order[0]),
        pin_value=0.0,
    )
    solved = phx.linalg.solve(
        pinned_system,
        pinned_rhs,
        policy=native_policy,
    )
    assert bool(solved.successful)
    np.testing.assert_allclose(solved.value[order[0]], 0, atol=_VOLUME_ATOL)
    physical_residual = diffusion.residual(
        solved.value[:2],
        solved.value[2:],
        jnp.eye(3),
        neumann,
        source=loads,
    )
    native_residual = pinned_system.operator.mv(solved.value) - pinned_rhs
    native_limit = (
        native_policy.tolerance.absolute
        + native_policy.tolerance.relative * jnp.linalg.norm(pinned_rhs)
    )
    assert float(jnp.linalg.norm(native_residual)) <= float(native_limit)
    # Source compatibility and physical flux cancellation recover the
    # removed row: r_pin=-sum(r_free). Hence ||r||₂≤sqrt(n)||r_free||₂.
    np.testing.assert_allclose(jnp.sum(loads), 0, atol=0)
    np.testing.assert_allclose(jnp.sum(physical_residual), 0, atol=_QUOTIENT_ATOL)
    free_rows = np.arange(physical_residual.size) != order[0]
    np.testing.assert_allclose(
        physical_residual[free_rows], native_residual[free_rows], atol=_VOLUME_ATOL
    )
    assert float(jnp.linalg.norm(physical_residual)) <= float(
        jnp.sqrt(physical_residual.size) * native_limit
    )
    # Two parallel half-cell paths each have conductance area/distance=2.
    # The independent discrete balance is 4*delta=1/8, so delta=1/32;
    # integrating the authored piecewise quadratic gives delta=1/64.
    # Their truncation difference is exactly 1/64, not an arbitrary .05.
    delta = float(solved.value[order[1]] - solved.value[order[0]])
    assert delta == pytest.approx(1 / 32, rel=0, abs=_OPERATOR_ATOL)
    assert abs(delta - 1 / 64) <= 1 / 64 + _OPERATOR_ATOL


def test_periodic_polyhedral_trace_reconstruction_residual_and_adjoint(
    periodic_fv_case: _PeriodicFVScenario,
    scalar_fv_evolution: _ScalarFVEvolution,
) -> None:
    prepared, diffusion = periodic_fv_case.prepared, periodic_fv_case.diffusion
    # The operator assertion requires solve accuracy below the default stopping rule.
    native_policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.ConjugateGradient(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=_VOLUME_ATOL, absolute=_VOLUME_ATOL
        ),
    )
    varying, actual_histories = (
        scalar_fv_evolution.varying,
        scalar_fv_evolution.trace_histories,
    )
    trace_boundary = HybridDiffusionBoundary(prepared)
    trace_system, cell_rhs_action, trace_offset = (
        diffusion.trace_reconstruction_operators(
            jnp.eye(3),
            trace_boundary,
        )
    )
    trace_prepared = phx.linalg.prepare(
        trace_system,
        native_policy,
    )
    for cell_values in (
        varying[:, 0],
        *(value[:, 0] for value in actual_histories),
        jnp.ones(prepared.cell_count),
    ):
        actual_rhs = cell_rhs_action.mv(cell_values) + trace_offset
        trace_solved = phx.linalg.solve(
            trace_prepared,
            actual_rhs,
        )
        assert bool(trace_solved.successful)
        local_flux = diffusion.local_fluxes(cell_values, trace_solved.value, jnp.eye(3))
        physical_face_residual = trace_boundary.face_residual(
            trace_solved.value,
            diffusion.continuity_residual(local_flux),
        )
        native_face_residual = trace_system.operator.mv(trace_solved.value) - actual_rhs
        np.testing.assert_allclose(
            physical_face_residual, native_face_residual, atol=_VOLUME_ATOL
        )
        native_limit = (
            native_policy.tolerance.absolute
            + native_policy.tolerance.relative * jnp.linalg.norm(actual_rhs)
        )
        assert float(jnp.linalg.norm(physical_face_residual)) <= float(native_limit)
    one = jnp.ones(diffusion.face_count)
    np.testing.assert_allclose(
        trace_system.operator.mv(one),
        cell_rhs_action.mv(jnp.ones(prepared.cell_count)) + trace_offset,
        atol=_VOLUME_ATOL,
    )
    # A constant cell field has exactly unit traces. Apply the existing
    # operator-reproduction regression allowance, not a 5-percent escape.
    np.testing.assert_allclose(trace_solved.value, one, rtol=0, atol=_OPERATOR_ATOL)
    face_cotangent = jnp.linspace(-1.0, 1.0, diffusion.face_count)
    cell_direction = jnp.linspace(0.2, 0.7, prepared.cell_count)
    np.testing.assert_allclose(
        jnp.vdot(face_cotangent, cell_rhs_action.mv(cell_direction)),
        jnp.vdot(cell_rhs_action.transpose_mv(face_cotangent), cell_direction),
        atol=_VOLUME_ATOL,
    )


def test_periodic_polyhedral_vector_flux_ledger_and_adjoint(
    periodic_fv_case: _PeriodicFVScenario,
    vector_fv_evolution: _VectorFVEvolution,
) -> None:
    euler, vector_prepared, vector_dynamics = (
        vector_fv_evolution.system,
        vector_fv_evolution.prepared,
        vector_fv_evolution.dynamics,
    )
    vector_state, action = vector_fv_evolution.initial_state, periodic_fv_case.action
    np.testing.assert_allclose(
        vector_dynamics(jnp.asarray(0.0), vector_state), 0, atol=_VOLUME_ATOL
    )
    action_matrix = euler.proper_isometry_state_matrix(jnp.asarray(action[:3, :3]))
    np.testing.assert_allclose(action_matrix[0], (1.0, 0.0, 0.0, 0.0, 0.0), atol=0)
    np.testing.assert_allclose(action_matrix[-1], (0.0, 0.0, 0.0, 0.0, 1.0), atol=0)
    np.testing.assert_allclose(action_matrix[1:4, 1:4], action[:3, :3], atol=0)
    flux, _ = vector_dynamics.face_fluxes(jnp.asarray(0.0), vector_state)
    face_active = vector_dynamics.discretization.face_block.active_mask
    expected = vector_dynamics._content_rate_from_fluxes(flux, active_mask=face_active)
    block = vector_dynamics.stage_rate_block_templates[0].with_flux_rate(
        flux * vector_prepared.face_measures[:, None],
    )
    ledger = ConservationStageLedger(
        [block],
        jnp.zeros_like(vector_state),
        jnp.ones((vector_prepared.cell_count,), dtype=jnp.bool_),
        geometry_family_id=vector_prepared.geometry_id,
        geometry_layout_id=vector_prepared.topology_id,
        geometry_version=jnp.asarray(0),
        evidence_policy_id=vector_prepared.geometry_id,
        evidence_version=jnp.asarray(0),
        topology_epoch_id=vector_prepared.topology_id,
    )
    np.testing.assert_allclose(ledger.scatter_content_rate(), expected, atol=_VOLUME_ATOL)
    source_sum, boundary_sum, net_sum = ledger.conservation_sums()
    np.testing.assert_allclose(
        net_sum,
        source_sum - boundary_sum + ledger.frame_exchange_sum(),
        atol=_VOLUME_ATOL,
    )
    raw_direction = (
        jnp.arange(flux.size, dtype=jnp.float64).reshape(flux.shape) / flux.size
    )
    cell_cotangent = (
        jnp.arange(vector_state.size, dtype=jnp.float64).reshape(vector_state.shape)
        / vector_state.size
    )
    scatter = lambda values: vector_dynamics._content_rate_from_fluxes(
        values, active_mask=face_active
    )
    _, tangent = jax.jvp(scatter, (flux,), (raw_direction,))
    _, pullback = jax.vjp(scatter, flux)
    np.testing.assert_allclose(
        jnp.vdot(cell_cotangent, tangent),
        jnp.vdot(pullback(cell_cotangent)[0], raw_direction),
        atol=_VOLUME_ATOL,
    )


def test_periodic_polyhedral_publication_identity(
    periodic_fv_case: _PeriodicFVScenario,
    periodic_fv_publication: CellMeshingResult,
) -> None:
    publication, construction = periodic_fv_publication, periodic_fv_case.construction
    original_source = periodic_fv_case.original_source
    assert publication.certification is not None and publication.certification.passed
    np.testing.assert_array_equal(
        publication.mesh.entity_set(3).entity_ids,
        construction.mesh.entity_set(3).entity_ids,
    )
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(publication.geometry),
    ) as budget:
        assert publication.geometry.source_coordinates(budget=budget) == original_source


@pytest.mark.parametrize(
    "field, limit_field, multiplier, offset",
    [
        ("source_expression_work_units", "maximum_work_units", 0, -1),
        ("source_expression_work_units", "maximum_work_units", 1, 1),
        ("source_expression_peak_bytes", "maximum_scratch_bytes", 0, -1),
        ("source_expression_peak_bytes", "maximum_scratch_bytes", 1, 1),
    ],
    ids=["negative-work", "excess-work", "negative-bytes", "excess-bytes"],
)
def test_periodic_polyhedral_certificate_refuses_expression_quantity(
    periodic_fv_publication: CellMeshingResult,
    field: str,
    limit_field: str,
    multiplier: int,
    offset: int,
) -> None:
    publication = periodic_fv_publication
    original = publication.certification
    assert original is not None and original.embedding is not None
    maximum = getattr(original.request.limits, limit_field)
    with pytest.raises(ValueError, match=rf"MeshCertificationReport\.embedding\.{field}"):
        altered = _corrupt_record_path(
            original, ("embedding", field), multiplier * maximum + offset
        )
        require_same_source_evidence(
            altered,
            original,
            publication.mesh,
            publication.geometry,
            publication.audit,
            limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )


@pytest.mark.parametrize(
    "path, diagnostic",
    [
        (("embedding", "certificate_id"), r"GlobalEmbeddingCertificate\.stored_fields"),
        (
            ("embedding", "binding", "binding_id"),
            r"stored_fields\.MeshCertificateBinding",
        ),
        (("embedding", "binding", "limits_id"), r"stored_fields\.MeshCertificateBinding"),
        (("request", "request_id"), r"MeshCertificationInputs\.request_id"),
        (("report_id",), r"stored_fields\.MeshCertificationReport"),
    ],
    ids=["certificate-id", "binding-id", "limits-id", "request-id", "report-id"],
)
def test_periodic_polyhedral_certificate_refuses_stored_identity(
    periodic_fv_publication: CellMeshingResult,
    path: tuple[str, ...],
    diagnostic: str,
) -> None:
    publication = periodic_fv_publication
    original = publication.certification
    assert original is not None and original.embedding is not None
    with pytest.raises(ValueError, match=diagnostic):
        altered = _corrupt_record_path(original, path, "0" * 64)
        require_same_source_evidence(
            altered,
            original,
            publication.mesh,
            publication.geometry,
            publication.audit,
            limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )


def test_periodic_polyhedral_certificate_refuses_cell_count(
    periodic_fv_publication: CellMeshingResult,
) -> None:
    publication = periodic_fv_publication
    original = publication.certification
    assert original is not None and original.embedding is not None
    with pytest.raises(ValueError, match=r"GlobalEmbeddingCertificate\.stored_fields"):
        altered = _corrupt_record_path(
            original, ("embedding", "cell_count"), original.embedding.cell_count + 1
        )
        require_same_source_evidence(
            altered,
            original,
            publication.mesh,
            publication.geometry,
            publication.audit,
            limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )


def test_periodic_polyhedral_certificate_refuses_outcome_identity(
    periodic_fv_publication: CellMeshingResult,
) -> None:
    publication = periodic_fv_publication
    original = publication.certification
    assert original is not None and original.embedding is not None
    index = next(
        i
        for i, outcome in enumerate(original.outcomes)
        if outcome.check == "global_embedding"
    )
    with pytest.raises(ValueError, match=r"stored_fields\.MeshCertificationOutcome"):
        altered = _corrupt_record_path(
            original, ("outcomes", index, "outcome_id"), "0" * 64
        )
        require_same_source_evidence(
            altered,
            original,
            publication.mesh,
            publication.geometry,
            publication.audit,
            limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )


def test_periodic_polyhedral_certificate_refuses_geometry_bits(
    periodic_fv_publication: CellMeshingResult,
) -> None:
    publication = periodic_fv_publication
    original = publication.certification
    assert original is not None and original.embedding is not None
    coordinates = np.asarray(publication.geometry.coordinates).copy()
    coordinates.view(np.uint64).reshape(-1)[0] ^= np.uint64(1)
    with pytest.raises(ValueError, match=r"MeshCertificateBinding\.geometry"):
        altered_geometry = _corrupt_record_path(
            publication.geometry, ("coordinates",), jnp.asarray(coordinates)
        )
        require_same_source_evidence(
            original,
            original,
            publication.mesh,
            altered_geometry,
            publication.audit,
            limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )


def test_public_record_restoration_preserves_identity_and_frozen_guard() -> None:
    register_meshing_source_artifacts()
    original = NativePolyhedralSchedule()
    restored = _corrupt_record_path(original, ("maximum_cells",), original.maximum_cells)
    assert restored.schedule_id == original.schedule_id
    assert restored.maximum_cells == original.maximum_cells
    with pytest.raises(AttributeError, match="maximum_cells"):
        restored.maximum_cells = 1


def test_public_record_restoration_refuses_unknown_field() -> None:
    register_meshing_source_artifacts()
    with pytest.raises(ValueError, match=r"corruption\.field"):
        _corrupt_record_path(NativePolyhedralSchedule(), ("not_a_field",), 1)


def test_public_record_restoration_refuses_invalid_capacity() -> None:
    register_meshing_source_artifacts()
    with pytest.raises(ValueError, match="maximum_cells"):
        altered = _corrupt_record_path(NativePolyhedralSchedule(), ("maximum_cells",), 0)
        validate_meshing_source_closure(altered, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)


def test_public_record_fault_changes_only_selected_field() -> None:
    original = NativePolyhedralSchedule()
    maximum_cells = original.maximum_cells
    altered = _corrupt_record_path(original, ("maximum_cells",), maximum_cells + 1)
    assert altered.maximum_cells == maximum_cells + 1
    assert original.maximum_cells == maximum_cells
    for field in fields(original):
        if field.name != "maximum_cells":
            assert getattr(altered, field.name) == getattr(original, field.name)


def _periodic_fv_field_declaration(
    publication: CellMeshingResult,
    restored_owner: phx.discretization.UnstructuredFiniteVolumeDiscretization,
    euler: phx.equations.EulerSystem,
    arrays: dict[str, Array],
) -> MeshingFiniteVolumeFieldDeclaration:
    binding = prepare_meshing_field_index_binding(
        publication,
        "field/global_coefficient_ids",
        field_space=restored_owner.cell_space,
    )
    density_unit = phx.units.derived_unit(
        "kg/m^3", ((phx.units.KILOGRAM, 1), (phx.units.METER, -3))
    )
    momentum_unit = phx.units.derived_unit(
        "kg/(m^2*s)",
        ((phx.units.KILOGRAM, 1), (phx.units.METER, -2), (phx.units.SECOND, -1)),
    )
    units = (density_unit, momentum_unit, momentum_unit, momentum_unit, phx.units.PASCAL)
    roles = tuple(
        MeshingFieldStateRole(
            name,
            "state",
            "coefficients"
            if name == "vector_state"
            else "stage-history"
            if name.endswith("history")
            else "material-history",
            tuple(value.shape),
            np.dtype(value.dtype),
            4,
            units
            if name in ("vector_state", "vector_history")
            else (density_unit,)
            if name == "density"
            else (phx.units.ONE,),
            index_binding=binding,
        )
        for name, value in arrays.items()
        if name != "state_epoch"
    ) + (
        MeshingFieldStateRole(
            "state_epoch", "state", "state-epoch", (), np.dtype("int64"), 4
        ),
    )
    reconstruction = phx.discretization.PiecewiseConstantReconstruction()
    declaration = MeshingFiniteVolumeFieldDeclaration(
        part_name="generation",
        topology_id=publication.mesh.topology_id,
        geometry_layout_id=publication.geometry.geometry_layout_id,
        field_space_ids=frozendict({"state": restored_owner.cell_space.field_space_id}),
        value_units=frozendict({"state": units}),
        maximum_derivative_orders=frozendict({"state": 0}),
        branch_policy=BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=roles,
        history_policy="material-history",
        numeric_version=restored_owner.numeric_version,
        finite_volume_field_name="state",
        finite_volume_component_names=euler.component_names,
        finite_volume_coordinate_policy="accepted-map",
        reconstruction_policy=reconstruction,
        reconstruction_policy_id=reconstruction.plan_id,
        finite_volume_system=euler,
        finite_volume_flux_policy=phx.discretization.RusanovFluxPlan(),
        finite_volume_boundaries=frozendict(
            {
                name: phx.discretization.ExtrapolationBoundary()
                for name in restored_owner.boundary_patch_names
            }
        ),
    )
    return declaration


def test_periodic_polyhedral_cold_archive_restores_fields_and_quotient(
    periodic_fv_case: _PeriodicFVScenario,
    scalar_fv_evolution: _ScalarFVEvolution,
    vector_fv_evolution: _VectorFVEvolution,
    periodic_fv_publication: CellMeshingResult,
    tmp_path: Path,
) -> None:
    publication, prepared = periodic_fv_publication, periodic_fv_case.prepared
    generation_source, specification, options = (
        periodic_fv_case.generation_source,
        periodic_fv_case.specification,
        periodic_fv_case.options,
    )
    euler, vector_prepared = vector_fv_evolution.system, vector_fv_evolution.prepared
    vector_state, vector_history = vector_fv_evolution.state, vector_fv_evolution.history
    state, history = scalar_fv_evolution.state, scalar_fv_evolution.history
    manufactured_history = scalar_fv_evolution.manufactured_history
    restored_owner = UnstructuredFiniteVolumePlan.from_cell_mesh(
        publication.mesh,
        component_names=euler.component_names,
    ).prepare(cell_geometry=publication.geometry)
    np.testing.assert_array_equal(
        restored_owner.cell_centers, vector_prepared.cell_centers
    )
    arrays = {
        "vector_state": vector_state,
        "vector_history": jnp.moveaxis(jnp.stack(vector_history), 0, 1),
        "scalar_state": state,
        "scalar_history": jnp.moveaxis(jnp.stack(history), 0, 1),
        "density": jnp.ones((prepared.cell_count,)),
        "state_epoch": jnp.asarray(4, dtype=jnp.int64),
    }
    if manufactured_history:
        arrays["manufactured_history"] = jnp.moveaxis(
            jnp.stack(manufactured_history), 0, 1
        )
    declaration = _periodic_fv_field_declaration(
        publication, restored_owner, euler, arrays
    )
    report = publication.certification
    assert report is not None and report.passed
    receipt = write_meshing_source_closure(
        tmp_path / "periodic-fv",
        {
            "certification_inputs": report.request,
            "report": report,
            "associations": publication.associations,
            "generation_part": M.MeshPart("generation", publication),
            "generation_source": generation_source,
            "generation_specification": specification,
            "generation_options": options,
            "accepted_data": {"fields": arrays},
            "field_declarations": {"generation": declaration},
        },
    )
    cold = run_on_forced_devices(
        "tests._support.distributed_meshing.cold_lifecycle",
        "periodic_finite_volume",
        (str(tmp_path / "periodic-fv"), receipt.content_id),
        devices=1,
        env={},
    )
    assert cold["old_report_alias"] and cold["current_report_alias"]
    np.testing.assert_allclose(record_array(cold["local_fluxes"]), 0, atol=_QUOTIENT_ATOL)
    np.testing.assert_array_equal(
        record_array(cold["scalar_history"]),
        np.ones((cold["cell_count"], 4, 1), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        record_array(cold["density"]), np.ones(cold["cell_count"], dtype=np.float64)
    )
    assert cold["quotient_face_count"] < cold["physical_face_count"]
    np.testing.assert_allclose(record_array(cold["vector_rate"]), 0, atol=_VOLUME_ATOL)
    cold_vector_state = record_array(cold["vector_state"])
    cold_vector_history = record_array(cold["vector_history"])
    np.testing.assert_allclose(
        cold_vector_history,
        np.broadcast_to(cold_vector_state[:, None, :], cold_vector_history.shape),
        atol=_VOLUME_ATOL,
    )


def test_polyhedral_schedule_normalizes_controls_and_retains_identity() -> None:
    supplied_volumes = [0.25, 0.75]
    supplied_sites = [0]
    schedule = NativePolyhedralSchedule(
        target_site_volumes=supplied_volumes,
        protected_sites=supplied_sites,
    )
    supplied_volumes[0] = 0.5
    supplied_sites.append(1)
    assert schedule.target_site_volumes == (0.25, 0.75)
    assert schedule.protected_sites == (0,)
    equivalent = NativePolyhedralSchedule(
        target_site_volumes=(0.25, 0.75), protected_sites=(0,)
    )
    assert schedule.schedule_id == equivalent.schedule_id
    assert schedule.schedule_id != NativePolyhedralSchedule().schedule_id
    with pytest.raises(AttributeError, match="maximum_cells"):
        schedule.maximum_cells = 1


@pytest.mark.parametrize(
    "capacities, field",
    [
        ((0, 1, 1, 1), "maximum_vertices"),
        ((1, 0, 1, 1), "maximum_cells"),
        ((1, 1, 0, 1), "maximum_work_units"),
        ((1, 1, 1, 0), "maximum_scratch_bytes"),
    ],
    ids=["vertices", "cells", "work", "scratch"],
)
def test_polyhedral_schedule_refuses_nonpositive_capacity(
    capacities: tuple[int, int, int, int],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        NativePolyhedralSchedule(
            maximum_vertices=capacities[0],
            maximum_cells=capacities[1],
            maximum_work_units=capacities[2],
            maximum_scratch_bytes=capacities[3],
        )


@pytest.mark.parametrize(
    "refinement, relocation, field",
    [(-1, 0, "maximum_refinement_steps"), (0, -1, "lloyd_steps")],
    ids=["refinement", "relocation"],
)
def test_polyhedral_schedule_refuses_negative_step_count(
    refinement: int,
    relocation: int,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        NativePolyhedralSchedule(
            maximum_refinement_steps=refinement, lloyd_steps=relocation
        )


@pytest.mark.parametrize(
    "volumes",
    [(0.0,), (-1.0,), (np.inf,), (np.nan,)],
    ids=["zero", "negative", "infinite", "nan"],
)
def test_polyhedral_schedule_refuses_invalid_volume_targets(
    volumes: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="target_site_volumes"):
        NativePolyhedralSchedule(target_site_volumes=volumes)


@pytest.mark.parametrize(
    "reason, category",
    [
        (
            PolyhedralGenerationReason.WORK_BUDGET,
            M.MeshingFailureCategory.RESOURCE_EXHAUSTED,
        ),
        (
            PolyhedralGenerationReason.FEATURE_CANDIDATE_BUDGET,
            M.MeshingFailureCategory.RESOURCE_EXHAUSTED,
        ),
        (
            PolyhedralGenerationReason.SOURCE_VERTEX_BUDGET,
            M.MeshingFailureCategory.RESOURCE_EXHAUSTED,
        ),
        (
            PolyhedralGenerationReason.PERIODIC_VOLUME_OPTIMIZATION_UNRESOLVED,
            M.MeshingFailureCategory.COMPLIANCE_FAILED,
        ),
        (
            PolyhedralGenerationReason.AMBIGUOUS_FEATURE_ANCESTRY,
            M.MeshingFailureCategory.COMPLIANCE_FAILED,
        ),
    ],
    ids=["work", "feature-candidates", "source-vertices", "volume", "feature-ancestry"],
)
def test_polyhedral_refusal_reason_retains_category_and_evidence(
    reason: PolyhedralGenerationReason,
    category: M.MeshingFailureCategory,
) -> None:
    evidence = {"covered_segments": 0.0}
    refusal = PolyhedralGenerationError(reason, entities=(3,), achieved=evidence)
    assert refusal.reason is reason
    assert refusal.reason.failure_category is category
    assert refusal.entities == (3,)
    assert refusal.achieved is evidence


@pytest.mark.parametrize(
    "cubes, regions, expected_interfaces",
    [
        ([(0, 0, 0)], None, 0),
        ([(0, 0, 0), (1, 0, 0)], None, 0),
        ([(0, 0, 0), (1, 0, 0)], [0, 1], 1),
        ([(0, 0, 0), (3, 0, 0)], None, 0),
    ],
    ids=["singleton", "joined", "material-interface", "disconnected"],
)
def test_voxel_fixture_oriented_union_volume_and_material_interfaces(
    cubes: Sequence[tuple[int, int, int]],
    regions: Sequence[int] | None,
    expected_interfaces: int,
) -> None:
    domain = voxel_plc(cubes, regions)
    volume = Fraction(0)
    interfaces = 0
    for polygon, (start, stop) in enumerate(
        zip(domain.polygon_offsets[:-1], domain.polygon_offsets[1:], strict=True)
    ):
        sides = domain.facet_regions[domain.polygon_facets[polygon]]
        if np.all(sides >= 0):
            interfaces += 1
            continue
        points = [
            tuple(Fraction(float(value)) for value in point)
            for point in domain.vertices[domain.polygon_vertices[start:stop]]
        ]
        for index in range(1, len(points) - 1):
            p, q, r = points[0], points[index], points[index + 1]
            cross = (
                q[1] * r[2] - q[2] * r[1],
                q[2] * r[0] - q[0] * r[2],
                q[0] * r[1] - q[1] * r[0],
            )
            volume += sum(p[axis] * cross[axis] for axis in range(3)) / 6
    # Divergence of x/3 integrates each oriented triangular boundary piece;
    # the independent union volume is the number of disjoint unit-cube interiors.
    assert volume == len(cubes)
    assert interfaces == expected_interfaces


def test_polyhedral_generator_refuses_foreign_source_before_native_scope() -> None:
    with pytest.raises(TypeError, match="complex_"):
        generate_polyhedral_volume(object(), scope=object())  # ty: ignore[invalid-argument-type]


def _source_star_affine(points: Array) -> Array:
    # Material interfaces in this extruded L are parallel to grad(u).
    return 1.0 + 3.0 * points[:, 2]


def test_source_star_fem_nonconvex_material_physical_solution() -> None:
    result = _publish(voxel_plc([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 0]))
    plan = prepare_polyhedral_source_star_fem(result)
    count = len(plan.cell_volumes)
    material = jnp.where(plan.space.mesh.coordinates[-count:, 0] > 1., 5., 2.)
    accepted = solve_polyhedral_source_star_diffusion(
        plan, jnp.zeros(count, dtype=jnp.float64),
        cell_material=material, exact_solution=_source_star_affine,
        repeats=3,
    )
    compiler = accepted.compiler
    assert compiler["kernel"] == "_solve_source_star_problem"
    assert compiler["compilation_count"] == 1
    assert all(
        type(value) in (int, float, str) or value is None
        or isinstance(value, tuple) and all(type(item) is float for item in value)
        for value in compiler.values()
    )
    warm_samples = compiler["warm_samples_seconds"]
    assert isinstance(warm_samples, tuple)
    assert len(warm_samples) == 3
    assert all(value >= 0 for value in warm_samples)
    retained_bytes = compiler["logical_retained_bytes"]
    assert type(retained_bytes) is int and retained_bytes > 0
    byte_fields = ("temporary_bytes", "output_bytes", "argument_bytes", "generated_code_bytes")
    if compiler["memory_status"] == "measured":
        assert all(
            type(value) is int and value >= 0
            for name in byte_fields for value in (compiler[name],)
        )
    else:
        assert compiler["memory_status"] == "unavailable"
        assert all(compiler[name] is None for name in byte_fields)
    assert float(accepted.l2_error) <= 1e-7
    assert float(accepted.maximum_free_residual) <= 1e-7
    assert float(accepted.conservation_defect) <= 1e-7
    np.testing.assert_allclose(
        accepted.gradients, np.broadcast_to([0., 0., 3.], accepted.gradients.shape),
        atol=1e-7, rtol=0,
    )
    np.testing.assert_allclose(
        accepted.vector_flux,
        -np.asarray(material)[np.asarray(plan.tetrahedron_parents), None]
        * np.asarray(accepted.gradients),
        atol=_VOLUME_ATOL, rtol=0,
    )
    loaded = solve_polyhedral_source_star_diffusion(
        plan, material, cell_material=material, exact_solution=_source_star_affine,
        seed=accepted.coefficients,
    )
    assert float(jnp.max(jnp.abs(loaded.coefficients-accepted.coefficients))) > 1e-6
    assert float(loaded.maximum_free_residual) <= 1e-7
    assert float(loaded.conservation_defect) <= 1e-7
    assert loaded.source_geometry_id == cell_geometry_id(result.geometry)
    assert loaded.prepared_id == plan.space.prepared_id
    assert loaded.runtime_id == plan.space.default_runtime.runtime_id
    assert plan.source_preparation_work_units > 0


def _source_star_halfcell(points: Array) -> Array:
    x = jnp.mod(points[:, 0], 1.)
    right = x - 0.5
    return jnp.where(
        x <= 0.5, 0.5*x*x-0.25*x, -0.5*right*right+0.25*right
    )


def test_source_star_fem_periodic_seams_and_physical_density() -> None:
    domain, constraints = periodic_cube_source()
    result = _publish(
        domain, sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.zeros(2, dtype=np.float64), periodic_constraints=constraints,
    )
    plan = prepare_polyhedral_source_star_fem(result)
    physical = np.asarray(plan.space.mesh.coordinates)
    triangles = {
        tuple(sorted(face))
        for tetrahedron in np.asarray(plan.tetrahedra)
        for face in itertools.combinations(tetrahedron.tolist(), 3)
    }
    for axis in range(3):
        seam_sides = []
        for side in (0., 1.):
            seam_sides.append({
                tuple(sorted(
                    tuple(np.delete(physical[index], axis)) for index in face
                ))
                for face in triangles
                if all(physical[index, axis] == side for index in face)
            })
        assert seam_sides[0] == seam_sides[1]
        assert seam_sides[0]
    centers = np.asarray(plan.space.mesh.coordinates)[np.asarray(plan.tetrahedra)].mean(axis=1)
    moments = np.zeros(len(plan.cell_volumes), dtype=np.float64)
    np.add.at(
        moments, np.asarray(plan.tetrahedron_parents),
        np.asarray(plan.tetrahedron_volumes)*centers[:, 0],
    )
    density = jnp.asarray(np.where(moments/np.asarray(plan.cell_volumes) < 0.5, 2., 4.))
    accepted = solve_polyhedral_source_star_diffusion(
        plan, density, cell_material=jnp.ones_like(density),
        exact_solution=_source_star_halfcell, periodic_density=True,
    )
    assert float(accepted.l2_error) <= 0.05
    assert float(jnp.max(jnp.abs(accepted.coefficients))) > 1e-6
    assert float(jnp.max(jnp.abs(accepted.vector_flux))) > 1e-6
    assert float(accepted.maximum_free_residual) <= 1e-7
    assert float(accepted.conservation_defect) <= 1e-7
    assert float(accepted.load_compatibility_residual) <= float(accepted.load_compatibility_bound)
    assert float(accepted.removed_density_mean) == pytest.approx(3.)
    quotient = np.asarray(plan.quotient_dofs)
    values = np.asarray(accepted.coefficients)
    for orbit in np.unique(quotient):
        assert np.all(values[quotient == orbit] == values[quotient == orbit][0])
    with pytest.raises(ValueError, match="physically incompatible mean"):
        solve_polyhedral_source_star_diffusion(
            plan, jnp.ones_like(density), cell_material=jnp.ones_like(density),
            exact_solution=_source_star_halfcell,
        )


def test_source_star_fem_transfers_history_and_cold_replays_accepted_seed() -> None:
    register_meshing_source_artifacts()
    domain = voxel_plc([(0, 0, 0)])
    source = _publish(
        domain, sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.zeros(2, dtype=np.float64),
    )
    before = prepare_polyhedral_source_star_fem(source)
    accepted = solve_polyhedral_source_star_diffusion(
        before, jnp.ones_like(before.cell_volumes),
        cell_material=jnp.ones_like(before.cell_volumes),
        exact_solution=_source_star_affine,
    )
    limits = M.MeshingLimits()
    associations = prepare_plc_associations(
        M.NativePlcSource(domain, "polyhedral-domain", "r1"), source, limits
    )
    adapted = M.execute_mesh_adaptation(M.prepare_mesh_adaptation(
        source,
        M.PolyhedralMeshAdaptation(
            M.PolyhedralAdaptationOperation.PLANE_SPLIT,
            planes=np.asarray([[0., 1., 0., 0.5]], dtype=np.float64),
        ),
        policy=M.MeshAdaptationPolicy(
            M.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
            limits=limits,
            association_transfer=associations.association_transfer,
            audit_policy=M.CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=M.CellMeshAuditDisposition.REJECT,
            ),
        ),
    ))
    target = prepare_polyhedral_source_star_fem(adapted.target)
    transfer = prepare_polyhedral_source_star_field_transfer(before, target)
    seed = transfer.primal_operator.mv(accepted.coefficients)
    history = tuple(
        transfer.primal_operator.mv(factor * accepted.coefficients)
        for factor in (0.5, 0.75, 1.)
    )
    np.testing.assert_allclose(
        transfer.primal_operator.mv(_source_star_affine(before.space.dof_maps[0].dof_coordinates)),
        _source_star_affine(target.space.dof_maps[0].dof_coordinates),
        rtol=0, atol=1e-12,
    )
    assert float(jnp.max(jnp.abs(seed-_source_star_affine(target.space.dof_maps[0].dof_coordinates)))) > 1e-6
    warm = solve_polyhedral_source_star_diffusion(
        target, jnp.ones_like(target.cell_volumes),
        cell_material=jnp.ones_like(target.cell_volumes),
        exact_solution=_source_star_affine, seed=seed, history=history,
    )
    # Cold continuation restores only the certified carrier and generic arrays;
    # no prepared matrix, reconstruction, or live compiler object is reused.
    bank = (adapted.target, seed, history)
    recipe = model_structure_recipe(bank)
    restored_carrier, restored_seed, restored_history = model_from_array_recipe(
        recipe, model_recipe_array_values(bank, recipe, prefix="source-star"),
        prefix="source-star",
    )
    cold_plan = prepare_polyhedral_source_star_fem(restored_carrier)
    cold = solve_polyhedral_source_star_diffusion(
        cold_plan, jnp.ones_like(cold_plan.cell_volumes),
        cell_material=jnp.ones_like(cold_plan.cell_volumes),
        exact_solution=_source_star_affine, seed=restored_seed,
        history=restored_history,
    )
    assert cold.plan_id == warm.plan_id
    assert cold.source_geometry_id == warm.source_geometry_id
    np.testing.assert_array_equal(cold.seed, seed)
    np.testing.assert_array_equal(jnp.stack(cold.history), jnp.stack(history))
    np.testing.assert_allclose(cold.coefficients, warm.coefficients, atol=1e-9, rtol=0)
    np.testing.assert_allclose(cold.vector_flux, warm.vector_flux, atol=1e-9, rtol=0)
    assert float(cold.maximum_free_residual) <= 1e-7
    assert float(cold.conservation_defect) <= 1e-7


def test_nonnested_vem_reconstruction_and_canonical_diagnostic_roles() -> None:
    import json

    from phydrax.lifecycle import _meshing_field_records as field_owner
    from tools.meshing_qualification.scenarios.polyhedral import (
        _prepare_polyhedral_accepted_epoch,
    )

    domain = voxel_plc([(0, 0, 0)])
    source = _publish(
        domain,
        sites=np.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)), dtype=np.float64),
        weights=np.zeros(2, dtype=np.float64),
    )
    target = _publish(
        domain,
        sites=np.asarray(((0.2, 0.5, 0.5), (0.5, 0.5, 0.5), (0.8, 0.5, 0.5)), dtype=np.float64),
        weights=np.zeros(3, dtype=np.float64),
    )
    for carrier in (source, target):
        provenance = dict(json.loads(carrier.provenance.content_json)["items"])
        assert provenance["specification"] == carrier.compliance.specification_id
        assert carrier.execution_evidence is not None
        assert provenance["plan"] == carrier.execution_evidence.owner_id
    before, after = prepare_polyhedral_archive_owners(source), prepare_polyhedral_archive_owners(target)
    fem = prepare_polyhedral_source_star_field_transfer(before.source_star, after.source_star)
    transfer = prepare_polyhedral_vem_reconstruction_transfer(
        before.vem, after.vem, before.source_star, after.source_star, fem
    )
    source_points = before.source_star.space.mesh.coordinates[:before.vem.dof_count]
    target_points = after.source_star.space.mesh.coordinates[:after.vem.dof_count]
    for values, expected in (
        (jnp.ones(before.vem.dof_count, dtype=jnp.float64), jnp.ones(after.vem.dof_count, dtype=jnp.float64)),
        (_source_star_affine(source_points), _source_star_affine(target_points)),
    ):
        np.testing.assert_allclose(transfer.primal_operator.mv(values), expected, atol=1e-12, rtol=0)
    primal = jnp.sin(jnp.arange(before.vem.dof_count, dtype=jnp.float64) + 0.3)
    dual = jnp.cos(jnp.arange(after.vem.dof_count, dtype=jnp.float64) + 0.7)
    pullback = transfer.dual_pullback_operator
    assert pullback is not None
    np.testing.assert_allclose(
        jnp.vdot(dual, transfer.primal_operator.mv(primal)),
        jnp.vdot(pullback.mv(dual), primal), atol=1e-12, rtol=0,
    )
    assert transfer.properties.semantics == "reconstruction"
    assert not transfer.properties.nested
    assert not transfer.properties.positivity_preserving
    assert not transfer.properties.conservative
    with pytest.raises(ValueError, match="actual canonical source-star"):
        prepare_polyhedral_vem_reconstruction_transfer(
            before.vem, after.vem, after.source_star, before.source_star, fem
        )
    vem_values = jnp.ones(before.vem.dof_count, dtype=jnp.float64)
    fem_values = _source_star_affine(before.source_star.space.dof_maps[0].dof_coordinates)
    inventory = jnp.ones_like(before.finite_volume.cell_volumes)
    epoch = _prepare_polyhedral_accepted_epoch(
        source,
        {
            "vem_solution": vem_values,
            "vem_solution_history": jnp.stack((vem_values,) * 3),
            "vem_analytic_nodal": vem_values,
            "fem_solution": fem_values,
            "fem_solution_history": jnp.stack((fem_values,) * 3),
            "inventory": inventory,
            "inventory_history": jnp.stack((inventory,) * 3),
            "previous_inventory": 0.9 * inventory,
        },
        0, owners=before,
    )
    declaration = epoch.declarations[0]
    assert isinstance(declaration, field_owner.MeshingPolyhedralFieldDeclaration)
    assert {role.entry_name for role in declaration.state_roles if role.role == "coefficients"} == {"vem_solution", "fem_solution"}
    analytic = next(role for role in declaration.state_roles if role.entry_name == "vem_analytic_nodal")
    assert analytic.role == "physical-values"
    assert declaration.normalization_reference_length is not None
    assert declaration.normalization_reference_length.unit_id == phx.units.METER.unit_id


def test_vem_kernel_stabilization_uses_gradient_rank_not_boundary_dof_count() -> None:
    points = np.asarray(
        [(row & 1, (row >> 1) & 1, (row >> 2) & 1) for row in range(8)],
        dtype=np.float64,
    )
    mesh = CellMesh.from_polyhedra(points, (POLYHEDRAL_CUBE_LOOPS,))
    prepared = prepare_polyhedral_h1_virtual_element_3d(mesh)
    mode = jnp.asarray((2 * points[:, 0] - 1) * (2 * points[:, 1] - 1))
    energy = float(jnp.vdot(mode, prepared.mv(mode)))
    # This mode is orthogonal to all four affine nodal columns and has norm²=8.
    # Affine reproduction forces ||G||_F² >= 3/2 on the unit cube, so the
    # mean nonzero consistency scale (trace/3) gives kernel energy at least 4.
    assert energy >= 4.0 - 1e-12


def _structured_polyhedral_patch(count: int, subdivide: bool) -> tuple[CellMesh, np.ndarray]:
    vertices: list[tuple[int, int, int]] = []
    index: dict[tuple[int, int, int], int] = {}
    cells: list[tuple[tuple[int, ...], ...]] = []
    centers = []
    segments = 2 if subdivide else 1
    for x, y, z in itertools.product(range(count), repeat=3):
        corners = np.asarray(
            [(2*x + 2*(row & 1), 2*y + 2*((row >> 1) & 1), 2*z + 2*((row >> 2) & 1))
             for row in range(8)],
            dtype=np.int64,
        )
        faces = []
        for face in POLYHEDRAL_CUBE_LOOPS:
            origin, first, _, last = corners[np.asarray(face)]
            du, dv = (first - origin) // segments, (last - origin) // segments
            for u, v in itertools.product(range(segments), repeat=2):
                loop = []
                for a, b in ((u, v), (u+1, v), (u+1, v+1), (u, v+1)):
                    point = origin + a*du + b*dv
                    key = (int(point[0]), int(point[1]), int(point[2]))
                    if key not in index:
                        index[key] = len(vertices)
                        vertices.append(key)
                    loop.append(index[key])
                faces.append(tuple(loop))
        cells.append(tuple(faces))
        centers.append(((x+0.5)/count, (y+0.5)/count, (z+0.5)/count))
    coordinates = np.asarray(vertices, dtype=np.float64) / (2*count)
    return CellMesh.from_polyhedra(coordinates, tuple(cells)), np.asarray(centers, dtype=np.float64)


@pytest.mark.parametrize("subdivide", (False, True), ids=("whole-facets", "facet-edge-subdivisions"))
def test_vem_nonconstant_poisson_converges_with_boundary_subdivisions(subdivide: bool) -> None:
    errors = []
    for count in (2, 4, 8):
        mesh, centers = _structured_polyhedral_patch(count, subdivide)
        prepared = prepare_polyhedral_h1_virtual_element_3d(mesh)
        points = np.asarray(mesh.coordinates)
        exact = np.prod(np.sin(np.pi * points), axis=1)
        half = 0.5/count
        # Exact cell averages of f=3*pi²*sin(pi*x)*sin(pi*y)*sin(pi*z).
        forcing = 3*np.pi**2 * np.prod(
            count*(np.cos(np.pi*(centers-half)) - np.cos(np.pi*(centers+half)))/np.pi,
            axis=1,
        )
        rhs_full = prepared.assemble_cellwise_constant_load(jnp.asarray(forcing))
        free = np.flatnonzero(np.all((points > 0) & (points < 1), axis=1))
        indices = jnp.asarray(free, dtype=jnp.int32)
        base = prepared.as_linear_operator()

        def action(value: Array) -> Array:
            return base.mv(jnp.zeros(len(points), dtype=jnp.float64).at[indices].set(value))[indices]

        space = phx.linalg.ArraySpace((len(free),), dtype=jnp.float64)
        operator = phx.linalg.FunctionLinearOperator(
            action, source=space, target=space,
            properties=phx.linalg.OperatorProperties(
                self_adjoint=True, positive_definite=True,
                evidence={"self_adjoint": "construction", "positive_definite": "construction"},
            ),
        )
        solved = phx.linalg.solve(
            phx.linalg.LinearSystem(operator), rhs_full[indices],
            policy=phx.linalg.LinearSolvePolicy(phx.linalg.ConjugateGradient()),
        )
        assert bool(jnp.all(solved.successful))
        values = np.zeros(len(points), dtype=np.float64)
        values[free] = np.asarray(solved.value)
        error = float(np.sqrt(np.mean((values-exact)**2)))
        errors.append(error)
        assert prepared.evidence.maximum_reproduction_defect <= 1e-10
        assert float(jnp.max(jnp.abs(operator.mv(solved.value)-rhs_full[indices]))) <= 1e-8
    assert errors[1] < 0.6*errors[0]
    assert errors[2] < 0.6*errors[1]


def test_periodic_power_operands_extend_the_actual_paid_generation_request() -> None:
    import pickle
    from copy import copy
    from dataclasses import replace

    from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
    from phydrax.meshing.qualification import execute_native_consumer
    from tools._meshing_cases import prepare_polyhedral_case
    from tools.meshing_qualification._records import qualification_failure

    source, request, options = prepare_polyhedral_case(
        "polyhedral-periodic-cube", 4, 20_000, 120.0, source=True
    )
    plan = phx.meshing.NativeMeshingProvider(options).plan(
        source, request, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    collector = CoordinateOperandCollector()
    generated = plan.execute(coordinate_operands=collector)
    operands = collector.prepared
    assert operands is not None
    earning_receipt = generated.execution_evidence
    assert earning_receipt is not None
    assert operands.earning_receipt is earning_receipt
    assert operands._power_witnesses
    with pytest.raises(TypeError, match="detach"):
        pickle.dumps(operands)
    with coordinate_reference_budget() as budget:
        with pytest.raises(ValueError, match="actual live native request"):
            operands.require(
                generated.mesh, generated.geometry, earning_receipt, budget,
                coordinate_contract=generated.coordinate_contract,
            )
        with pytest.raises(ValueError, match="exact paid earning receipt"):
            operands.require(
                generated.mesh, generated.geometry,
                copy(earning_receipt), budget,
                coordinate_contract=generated.coordinate_contract,
            )

    def prepare(budget: CoordinateEnclosureBudget) -> None:
        for identity, witness in operands._power_witnesses.items():
            assert budget.power_witness_cache[identity] is witness
        identity, witness = next(iter(operands._power_witnesses.items()))
        budget.power_witness_cache[identity] = replace(witness)
        with pytest.raises(ValueError, match="conflict.*exact power"):
            operands.require(
                generated.mesh, generated.geometry, earning_receipt, budget,
                coordinate_contract=generated.coordinate_contract,
            )
        budget.power_witness_cache[identity] = witness
        operands.require(
            generated.mesh, generated.geometry, earning_receipt, budget,
            coordinate_contract=generated.coordinate_contract,
        )
        owners = prepare_polyhedral_archive_owners(generated, budget=budget)
        assert owners.source_star.periodic
        assert owners.finite_volume.mesh is generated.mesh

    consumed = execute_native_consumer(
        generated, request.limits, prepare,
        failure_factory=qualification_failure, prepared_operands=operands,
        observe_managed_admission=True,
    )
    receipt = consumed.result.execution_evidence
    assert receipt is not None
    assert receipt.preparation_evidence is earning_receipt
    managed_peak = receipt.total_managed_admission_peak_bytes_upper
    assert managed_peak is not None
    assert int(np.asarray(managed_peak)) <= request.limits.maximum_scratch_bytes
    assert int(np.asarray(receipt.total_work_units)) <= request.limits.maximum_work_units
    assert int(np.asarray(receipt.total_geometry_queries)) <= request.limits.maximum_geometry_queries


def test_polyhedral_source_vertex_prefix_shortfall_is_a_typed_resource_refusal() -> None:
    from dataclasses import replace

    from tools._meshing_cases import prepare_polyhedral_case
    from tools.meshing_qualification._controls import (
        MeshingLimitControls,
        with_meshing_limits,
    )

    source, request, options = prepare_polyhedral_case(
        "polyhedral-material-L", 4, 20_000, 120.0, source=True
    )
    required = source.complex.vertices.shape[0]
    limits = replace(
        MeshingLimitControls.from_limits(request.limits),
        maximum_vertices=required - 1,
    ).to_limits()
    plan = M.NativeMeshingProvider(options).plan(
        source, with_meshing_limits(request, limits),
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    with pytest.raises(M.MeshingFailure) as caught:
        plan.execute()
    refusal = caught.value
    assert refusal.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert refusal.__cause__ is not None
    assert isinstance(refusal.__cause__, PolyhedralGenerationError)
    assert refusal.__cause__.reason is PolyhedralGenerationReason.SOURCE_VERTEX_BUDGET
    assert dict(refusal.evidence.achieved) == {
        "required_vertices": required, "maximum_vertices": required - 1,
    }


def test_coordinate_input_binding_retains_the_actual_named_constraint_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization._coordinate_operands import (
        coordinate_input_owner_fields,
        require_coordinate_input_owners,
    )
    from phydrax.meshing._adaptation import _empty_constraints

    mesh, _ = _structured_polyhedral_patch(1, False)
    constraints = _empty_constraints(mesh)
    arrays: list[Array | np.ndarray] = []
    containers: list[tuple[list[object], tuple[object, ...]]] = []
    with coordinate_reference_budget() as budget:
        snapshot = coordinate_input_owner_fields(
            (constraints,), budget, arrays=arrays, containers=containers
        )
        assert snapshot is not None
        root = next(record for record in snapshot if record[0] is constraints)
        assert root[1] is type(constraints)
        assert root[2] == ("_fields", *constraints._fields)
        assert all(
            root[3][index + 1] is value for index, value in enumerate(constraints)
        )
        assert require_coordinate_input_owners(snapshot, budget)
        with monkeypatch.context() as mutation:
            mutation.setattr(
                type(constraints), "_fields", tuple(reversed(constraints._fields))
            )
            assert not require_coordinate_input_owners(snapshot, budget)
        assert require_coordinate_input_owners(snapshot, budget)

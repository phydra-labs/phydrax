from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellMesh
from phydrax.geometry import Sphere
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.meshing._adaptation import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.meshing._association import (
    AssociationEntityClasses,
    AssociationPropagationError,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._association_composition import ComposedAssociationTransfer
from phydrax.meshing._association_routes import boundary_mask, target_dimension
from phydrax.meshing._bisection import BisectionCompatibility, BisectionHierarchy
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._implicit_association_transfer import ImplicitAssociationTransfer
from phydrax.meshing._lineage import (
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    VertexInterpolationStencil,
)
from phydrax.meshing._mapped_reference_association import (
    mapped_reference_associations,
    MappedReferenceAssociationTransfer,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._surface_association_transfer import SurfaceAssociationTransfer
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native import NativeMeshingProvider
from phydrax.meshing.providers._native_implicit import (
    normalize_analytic_implicit_source,
)
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_schedules import (
    NativeCurveSchedule,
    NativeSurfaceSchedule,
)
from phydrax.meshing.providers._native_sources import NativeImplicitSource
from tests.unit.meshing.test_native_mapped_hex_routes import _mapped_plc_tetra_source
from tests.unit.meshing.test_native_surface_routes import (
    _square_domain,
    _surface_specification,
)
from tests.unit.meshing.test_transition_acceptance import tiny_chart_source


_CONTRACT = SpatialCoordinateContract.si()


def _profile(source_id: str = "association-sphere") -> AnalyticImplicitProfile:
    geometry = Sphere((0.0, 0.0, 0.0), 1.0, feature_id=source_id).compile()
    return AnalyticImplicitProfile(
        geometry,
        _CONTRACT,
        source_id=source_id,
        source_revision="association-r1",
    )


def _implicit_surface_result(profile: AnalyticImplicitProfile) -> CellMeshingResult:
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.2,
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
        maximum_deviation=0.2,
    )
    options = NativeMeshingOptions("implicit_surface", implicit_policy=policy)
    return (
        NativeMeshingProvider(options)
        .plan(native_source, specification, coordinate_contract=_CONTRACT)
        .execute()
    )


def _identity_lineage(mesh: CellMesh) -> tuple[MeshLineage, VertexInterpolationStencil]:
    entities = tuple(
        EntityLineage(
            dimension,
            mesh.entity_set(dimension).entity_set_id,
            mesh.entity_set(dimension).entity_set_id,
            np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64),
            np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64),
            np.full(
                (mesh.entity_set(dimension).count,),
                EntityLineageKind.PRESERVED,
                dtype=np.int32,
            ),
        )
        for dimension in range(mesh.topological_dimension + 1)
    )
    lineage = MeshLineage(mesh.topology_id, mesh.topology_id, entities)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    stencil = VertexInterpolationStencil(
        mesh.entity_set(0).entity_set_id,
        mesh.entity_set(0).entity_set_id,
        vertex_ids,
        vertex_ids[:, None],
        np.ones((vertex_ids.size, 1), dtype=np.float64),
        np.ones((vertex_ids.size, 1), dtype=np.bool_),
        exact_weights=tuple((Fraction(1),) for _ in vertex_ids.tolist()),
    )
    return lineage, stencil


def test_implicit_sphere_association_survives_bisection_split_and_collapse() -> None:
    profile = _profile()
    source = _implicit_surface_result(profile)
    certification = source.certification
    if certification is None:
        raise AssertionError("Native implicit result must retain source certification.")
    native_source = certification.inputs.feature_source
    assert isinstance(native_source, NativeImplicitSource)
    assert native_source.authoring_profile is profile
    transfer = ImplicitAssociationTransfer(native_source)
    foreign = AnalyticImplicitProfile(
        profile.geometry,
        _CONTRACT,
        source_id="same-geometry-other-source",
        source_revision="other-r1",
    )
    with pytest.raises(ValueError):
        ImplicitAssociationTransfer(foreign).validate_source(source)
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
        association_transfer=transfer,
    )
    source_cells = np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(source_cells),
            policy=policy,
        )
    )
    target = refined.target
    for dimension in range(3):
        association = next(
            value
            for value in target.associations
            if target_dimension(target.mesh, value) == dimension
        )
        assert association.complete
        assert association.source_id == profile.source_id
        assert association.source_revision == profile.source_revision
        assert set(association.source_entity_ids) == {
            source.associations[0].source_entity_ids[0]
        }
        assert all(
            role is GeometrySourceEntityRole.FACET for role in association.source_roles
        )
    target_vertices = next(
        value
        for value in target.associations
        if target_dimension(target.mesh, value) == 0
    )
    created = np.setdiff1d(
        np.asarray(target.mesh.vertex_global_ids, dtype=np.int64),
        np.asarray(source.mesh.vertex_global_ids, dtype=np.int64),
    )
    created_rows = target_vertices.target_rows(created)
    assert created.size > 0
    assert np.all(np.asarray(target_vertices.parent_dimensions)[created_rows] >= 1)
    assert np.all(np.asarray(target_vertices.parent_ids)[created_rows] >= 0)
    hierarchy = refined.hierarchy
    if not isinstance(hierarchy, BisectionHierarchy):
        raise AssertionError("Bisection refinement must retain its bisection hierarchy.")
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(
                np.empty((0,), dtype=np.int64),
                np.asarray(target.mesh.entity_set(2).entity_ids, dtype=np.int64),
                hierarchy=hierarchy,
            ),
            policy=policy,
        )
    )
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    assert all(value.complete for value in coarsened.target.associations)
    assert all(
        value.source_id == profile.source_id
        and all(role is GeometrySourceEntityRole.FACET for role in value.source_roles)
        for value in coarsened.target.associations
    )


def test_implicit_collapse_into_born_vertex_follows_its_target_stratum() -> None:
    profile = _profile()
    source = _implicit_surface_result(profile)
    transfer = ImplicitAssociationTransfer(profile)
    source_points = np.asarray(source.mesh.coordinates, dtype=np.float64)
    source_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    born = int(np.max(source_ids)) + 1
    # A source vertex collapses into a vertex born during adaptation (a split point).
    target = CellMesh(
        np.stack(
            (
                0.5 * (source_points[0] + source_points[1]),
                source_points[1],
                source_points[2],
            )
        ),
        (CellBlock("surface", "triangle", np.asarray([[0, 1, 2]], dtype=np.int64)),),
        vertex_global_ids=np.asarray([born, source_ids[1], source_ids[2]]),
    )
    record = EntityLineage(
        0,
        source.mesh.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        source_ids[:1],
        np.asarray([born], dtype=np.int64),
        np.asarray([EntityLineageKind.COLLAPSED_INTO], dtype=np.int32),
    )
    lineage = MeshLineage(source.mesh.topology_id, target.topology_id, (record,))
    source_classes = transfer.classes(source)
    surface_classes = AssociationEntityClasses(
        np.full((3,), 2, dtype=np.int64),
        np.zeros((3,), dtype=np.int64),
        np.ones((3,), dtype=np.bool_),
        np.ones((3,), dtype=np.int64),
    )
    # The born vertex lies on the zero set, so the collapse keeps stratum closure.
    transfer._check_collapses(source, lineage, source_classes, target, (surface_classes,))
    # A zero-set vertex collapsing into a born region-interior vertex leaves its
    # stratum and stays refused.
    region_classes = surface_classes._replace(dimensions=np.full((3,), 3, dtype=np.int64))
    with pytest.raises(AssociationPropagationError):
        transfer._check_collapses(
            source, lineage, source_classes, target, (region_classes,)
        )


def _curve_rows(result: CellMeshingResult) -> dict[int, tuple[object, ...]]:
    return {
        int(identifier): (
            association.source_entity_ids[row],
            association.source_parameter_intervals[row],
            int(np.asarray(association.orientations)[row]),
        )
        for association in result.associations
        if association.association_kind is GeometryAssociationKind.CURVE
        for row, identifier in enumerate(
            np.asarray(association.target_global_ids, dtype=np.int64).tolist()
        )
    }


def test_trimmed_sheet_association_transfer_retains_patches_and_curve_intervals() -> None:
    domain = _square_domain()
    specification = _surface_specification(domain)
    options = NativeMeshingOptions(
        "parametric_surface",
        surface_schedule=NativeSurfaceSchedule(
            quality_angle_degrees=20.0,
            spacing_fraction=0.45,
            minimum_edge_samples=4,
            curve_schedule=NativeCurveSchedule(fidelity_samples=5),
        ),
    )
    source = (
        NativeMeshingProvider(options)
        .plan(domain, specification, coordinate_contract=_CONTRACT)
        .execute()
    )
    transfer = SurfaceAssociationTransfer(domain)
    transfer.validate_source(source)
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
        association_transfer=transfer,
    )
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(
                np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
            ),
            policy=policy,
        )
    )
    faces = next(
        value
        for value in refined.target.associations
        if target_dimension(refined.target.mesh, value) == 2
    )
    curves = next(
        value
        for value in refined.target.associations
        if value.association_kind is GeometryAssociationKind.CURVE
    )
    assert faces.complete
    assert curves.complete
    surface_source_ids = domain.entity_ids(
        np.full(faces.target_global_ids.size, 2, dtype=np.int64),
        np.zeros(faces.target_global_ids.size, dtype=np.int64),
    )
    assert set(faces.source_entity_ids) <= set(surface_source_ids)
    curve_source_ids = {
        identifier
        for association in source.associations
        if association.association_kind is GeometryAssociationKind.CURVE
        for identifier in association.source_entity_ids
    }
    assert set(curves.source_entity_ids) == curve_source_ids
    assert all(
        interval is not None and interval[0] < interval[1]
        for interval in curves.source_parameter_intervals
    )


def test_coarsened_curve_edges_merge_exact_source_intervals() -> None:
    domain, source = tiny_chart_source()
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
        association_transfer=SurfaceAssociationTransfer(domain),
    )
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(
                np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
            ),
            policy=policy,
        )
    )
    current, hierarchy = refined.target, refined.hierarchy
    assert isinstance(hierarchy, BisectionHierarchy)
    # Every source curve edge is split, so coarsening merges COARSENED_INTO halves.
    assert len(_curve_rows(current)) > len(_curve_rows(source))
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            current,
            MarkedMeshAdaptation(
                (),
                np.asarray(current.mesh.entity_set(2).entity_ids, dtype=np.int64),
                hierarchy=hierarchy,
            ),
            policy=policy,
        )
    )
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    # Merged half-edges recover the source's exact interval and orientation.
    assert _curve_rows(coarsened.target) == _curve_rows(source)


def test_mapped_reference_root_association_and_composed_namespaces() -> None:
    _source, domain = _mapped_plc_tetra_source()
    mesh, geometry = domain.reference_mesh, domain.source_geometry
    mapped = mapped_reference_associations(domain, mesh, geometry)
    mapped_transfer = MappedReferenceAssociationTransfer(domain)
    mapped_result = certify_cell_mesh(
        mesh,
        domain.coordinate_contract,
        geometry=geometry,
        associations=mapped,
    )
    mapped_transfer.validate_source(mapped_result)
    cell_association = mapped[0]
    assert cell_association.complete and cell_association.exact
    assert cell_association.source_entity_ids == (
        domain.image_entity_id(3, int(domain.root_cell_ids[0])),
    )

    profile = AnalyticImplicitProfile(
        Sphere((0.0, 0.0, 0.0), 3.0, feature_id="composed-implicit-region").compile(),
        domain.coordinate_contract,
        source_id="composed-implicit-region",
        source_revision="r1",
    )
    region_key = profile.source_entity_key_for_scope(
        3, int(profile.scope_entity_ids(3)[0])
    )
    implicit_cells = GeometryAssociation(
        GeometryAssociationKind.IMPLICIT,
        profile.source_id,
        profile.source_revision,
        mesh.entity_set(3).entity_set_id,
        np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64),
        (region_key[0],) * mesh.entity_set(3).count,
        np.zeros((mesh.entity_set(3).count,), dtype=np.float64),
        exact=True,
        source_dimensions=np.full((mesh.entity_set(3).count,), 3, dtype=np.int8),
        source_indices=np.zeros((mesh.entity_set(3).count,), dtype=np.int64),
        source_roles=(GeometrySourceEntityRole.REGION,) * mesh.entity_set(3).count,
        source_occurrence_paths=(region_key[1],) * mesh.entity_set(3).count,
    )
    composed_source = certify_cell_mesh(
        mesh,
        domain.coordinate_contract,
        geometry=geometry,
        associations=(*mapped, implicit_cells),
    )
    transfer = ComposedAssociationTransfer(
        (mapped_transfer, ImplicitAssociationTransfer(profile))
    )
    levels = transfer.classes(composed_source)
    assert levels[3].resolved.tolist() == [True]
    lineage, stencil = _identity_lineage(mesh)
    result = transfer.propagate(composed_source, lineage, mesh, vertex_stencil=stencil)
    assert {(value.association_kind, value.source_id) for value in result} == {
        (GeometryAssociationKind.MAPPED_REFERENCE, domain.source_id),
        (GeometryAssociationKind.IMPLICIT, profile.source_id),
    }
    implicit_associations = tuple(
        value
        for value in result
        if value.association_kind is GeometryAssociationKind.IMPLICIT
    )
    assert implicit_associations
    assert all(
        all(role is GeometrySourceEntityRole.REGION for role in value.source_roles)
        for value in implicit_associations
    )


def _outward_boundary_triangles(volume: CellMesh, surface: CellMesh) -> bool:
    """Whether the surface is the volume boundary with normals leaving the origin."""

    corners = np.asarray(surface.coordinates)[
        np.concatenate([np.asarray(block.vertices) for block in surface.blocks])
    ]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    outward = np.sum(normals * np.mean(corners, axis=1), axis=1) > 0.0
    boundary = np.count_nonzero(boundary_mask(volume, 2))
    return corners.shape[0] == boundary and bool(np.all(outward))


def test_implicit_volume_boundary_remap_orients_triangles_outward() -> None:
    profile = _profile("association-ball")
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.2,
        root_tolerance=1.0e-8,
    )
    namespace = normalize_analytic_implicit_source(profile, policy, _CONTRACT)
    scope = MeshingScope(
        namespace.source_id,
        namespace.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        namespace.scope_entity_set_id(2),
        namespace.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=0.8,
    )
    options = NativeMeshingOptions(
        "implicit_adaptive_tetrahedral",
        implicit_policy=policy,
        volume_schedule=NativeVolumeSchedule(),
    )
    source = (
        NativeMeshingProvider(options)
        .plan(namespace, specification, coordinate_contract=_CONTRACT)
        .execute()
    )
    certification = source.certification
    if certification is None:
        raise AssertionError("Native implicit volume must retain source certification.")
    authority = certification.inputs.feature_source
    if not isinstance(authority, NativeImplicitSource):
        raise AssertionError("Native implicit volume must retain its implicit source.")
    transfer = ImplicitAssociationTransfer(authority)
    lineage, _ = _identity_lineage(source.mesh)
    # The native volume carries simplicial incidence; the copy carries
    # vertex-sorted tetrahedral face rows. Both remap to outward triangles.
    mesh = source.mesh
    copy = CellMesh.from_tetrahedra(
        np.asarray(mesh.coordinates),
        np.concatenate([np.asarray(block.vertices) for block in mesh.blocks]),
        vertex_global_ids=np.asarray(mesh.vertex_global_ids),
        cell_global_ids=np.concatenate(
            [np.asarray(block.global_ids) for block in mesh.blocks]
        ),
        numeric_version=mesh.numeric_version,
    )
    for target in (copy, mesh):
        model = transfer.remap_boundary(source, lineage, target)
        assert model is not None
        assert _outward_boundary_triangles(target, model.mesh)

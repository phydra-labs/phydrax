from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, TYPE_CHECKING

import equinox as eqx
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.geometry import Sphere
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryFaceImageWitness,
    SourceBoundaryImagePullback,
)
from phydrax.geometry._source_fidelity import (
    validate_source_fidelity_witness,
    ValidatedSourceFidelityWitness,
)
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.geometry.implicit._chart_restriction import (
    certify_implicit_chart_restriction,
    ImplicitChartPremise,
)
from phydrax.meshing._adaptation import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.meshing._association import (
    _target_piece_sort_key,
    AssociationEntityClasses,
    AssociationPropagationError,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
    SourceBoundaryTargetPieceRow,
)
from phydrax.meshing._association_composition import ComposedAssociationTransfer
from phydrax.meshing._association_routes import (
    boundary_mask,
    entity_rows,
    target_dimension,
)
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
from phydrax.meshing._surface_association_transfer import (
    _curve_edges_with_root_ancestry,
    _simplex_edges,
    _target_edge_parameter,
    SurfaceAssociationTransfer,
)
from phydrax.meshing._topology_edit import entity_keys, key_rows
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
from tests.unit.geometry.test_implicit_source_charts import _retarget_implicit_witness
from tests.unit.interchange.test_native_cad_archive import _worker_oracle
from tests.unit.meshing.test_native_mapped_hex_routes import _mapped_plc_tetra_source
from tests.unit.meshing.test_native_surface_routes import (
    _square_domain,
    _surface_specification,
)
from tests.unit.meshing.test_transition_acceptance import tiny_chart_source


if TYPE_CHECKING:
    from phydrax.meshing._layer_core_association_transfer import (
        NativeLayerCoreAssociationTransfer,
    )
    from phydrax.meshing._mixed_adaptation import MixedLayerColumns
    from phydrax.meshing.providers._native_sources import NativeLayerCoreSource

type _NativeLayerCoreFixture = tuple[
    NativeLayerCoreSource,
    VolumeMeshingSpec,
    CellMeshingResult,
    NativeLayerCoreAssociationTransfer,
    MixedLayerColumns,
]


_CONTRACT = SpatialCoordinateContract.si()


def _profile(source_id: str = "association-sphere") -> AnalyticImplicitProfile:
    geometry = Sphere((0.0, 0.0, 0.0), 1.0, feature_id=source_id).compile()
    return AnalyticImplicitProfile(
        geometry,
        _CONTRACT,
        source_id=source_id,
        source_revision="association-r1",
    )


def _implicit_surface_result(
    profile: AnalyticImplicitProfile, *, relative_cell_diameter: float = 0.2
) -> CellMeshingResult:
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=relative_cell_diameter,
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
    # Coarsening IDs belong to the refined target, not the pre-refinement source.
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            target,
            MarkedMeshAdaptation(
                np.empty((0,), dtype=np.int64),
                np.asarray(target.mesh.entity_set(2).entity_ids, dtype=np.int64),
                hierarchy=hierarchy,
            ),
            policy=policy,
            prior_transition=refined.transition,
        )
    )
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    assert all(value.complete for value in coarsened.target.associations)
    assert all(
        value.source_id == profile.source_id
        and all(role is GeometrySourceEntityRole.FACET for role in value.source_roles)
        for value in coarsened.target.associations
    )


def test_implicit_bisection_inherits_parent_chart_bounds_by_restriction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile("restriction-sphere")
    source = _implicit_surface_result(profile, relative_cell_diameter=0.4)
    certification = source.certification
    assert certification is not None and certification.fidelity is not None
    parent = certification.fidelity
    assert parent.status == "certified" and parent.chart_rows is not None
    native_source = certification.inputs.feature_source
    assert isinstance(native_source, NativeImplicitSource)
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
        association_transfer=ImplicitAssociationTransfer(native_source),
    )
    from phydrax.geometry.implicit import _source_charts

    parent_witness = parent.witness
    assert isinstance(parent_witness, SourceBoundaryChartWitness)
    build_face = _source_charts._build_face
    proved_witnesses: list[SourceBoundaryChartWitness] = []

    def recorded_proof(*args: Any, **kwargs: Any) -> Any:
        proved_witnesses.append(args[1])
        return build_face(*args, **kwargs)

    # Since 36467b364 the parent proof is replayed from scratch before any
    # operation-scoped reuse, so the parent's own rows are proved once more.
    # An exact chart restriction must not prove any target row.
    with monkeypatch.context() as reuse:
        reuse.setattr(_source_charts, "_build_face", recorded_proof)
        refined = execute_mesh_adaptation(
            prepare_mesh_adaptation(
                source,
                MarkedMeshAdaptation(
                    np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
                ),
                policy=policy,
            )
        )
    assert all(
        np.array_equal(
            np.asarray(witness.cell_global_ids),
            np.asarray(parent_witness.cell_global_ids),
        )
        and witness.source_chart_ids == parent_witness.source_chart_ids
        for witness in proved_witnesses
    )
    target = refined.target
    assert target.certification is not None
    assert target.certification.embedding.nested_restriction is not None
    inherited = target.certification.fidelity
    assert inherited is not None and inherited.chart_restriction is not None
    evidence = inherited.chart_restriction
    assert evidence.parent_certificate_id == parent.certificate_id
    assert evidence.target_row_count > evidence.parent_row_count
    inputs = target.certification.inputs
    assert inputs.fidelity_tolerance is not None
    # Independent cold replay is separately collected below, so the large target
    # is not retained while a redundant in-process parent proof is recomputed.
    assert inherited.status == "certified"

    # Negative: rows that do not tile their parent rows inherit nothing.
    witness = inputs.fidelity_witness
    assert isinstance(witness, SourceBoundaryChartWitness)
    premise = inherited.chart_premise
    assert isinstance(premise, ImplicitChartPremise)
    partial = _retarget_implicit_witness(
        witness,
        target.mesh,
        target.geometry,
        np.asarray(witness.cell_global_ids, dtype=np.int64),
        rows=tuple(range(len(witness.source_chart_ids) - 1)),
    )
    assert (
        certify_implicit_chart_restriction(
            target.mesh, target.geometry, partial, parent, premise
        )
        is None
    )
    overlapping = _retarget_implicit_witness(
        witness,
        target.mesh,
        target.geometry,
        np.asarray(witness.cell_global_ids, dtype=np.int64),
        rows=(0, 0, *range(2, len(witness.source_chart_ids))),
    )
    assert (
        certify_implicit_chart_restriction(
            target.mesh, target.geometry, overlapping, parent, premise
        )
        is None
    )
    elements, routes, coordinates = target.geometry.resolve(target.mesh)
    changed_coordinates = np.array(coordinates, copy=True)
    changed_coordinates[0, 0] = np.nextafter(changed_coordinates[0, 0], np.inf)
    # A source-mesh binding needs a source coordinate map (6bffc1354); the
    # nudged geometry is a plain coordinate field.
    changed_geometry = CellGeometrySpec(
        {
            block.name: element
            for block, element in zip(target.mesh.blocks, elements, strict=True)
        },
        {
            block.name: route
            for block, route in zip(target.mesh.blocks, routes, strict=True)
        },
        changed_coordinates,
    )
    with pytest.raises(ValueError, match="bound|geometry|inputs"):
        certify_implicit_chart_restriction(
            target.mesh, changed_geometry, witness, parent, premise
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


def test_curve_edge_parameter_uses_exact_rational_stencil() -> None:
    stencil = VertexInterpolationStencil(
        "source-vertices",
        "target-vertices",
        np.asarray((30,)),
        np.asarray(((10, 20),)),
        np.asarray(((2.0 / 3.0, 1.0 / 3.0),)),
        np.asarray(((True, True),)),
        exact_weights=((Fraction(2, 3), Fraction(1, 3)),),
    )
    assert _target_edge_parameter(30, (10, 20), stencil) == Fraction(1, 3)
    with pytest.raises(ValueError, match="source edge"):
        _target_edge_parameter(30, (10, 40), stencil)


def test_curve_edge_incidence_supports_volume_cells() -> None:
    _source, domain = _mapped_plc_tetra_source()
    edges = _simplex_edges(domain.reference_mesh)
    assert edges.shape == (domain.reference_mesh.entity_set(1).count, 2)


def test_curve_edge_replay_retains_original_root_parent() -> None:
    _domain, source = tiny_chart_source()
    original = next(
        value
        for value in source.associations
        if value.association_kind is GeometryAssociationKind.CURVE
        and value.target_entity_set_id == source.mesh.entity_set(1).entity_set_id
    )
    ids = np.asarray(original.target_global_ids, dtype=np.int64)
    parent = GeometryAssociation(
        original.association_kind,
        original.source_id,
        original.source_revision,
        original.target_entity_set_id,
        ids,
        original.source_entity_ids,
        np.asarray(original.residuals),
        source_dimensions=np.asarray(original.source_dimensions),
        source_indices=np.asarray(original.source_indices),
        source_roles=original.source_roles,
        source_occurrence_paths=original.source_occurrence_paths,
        source_parameter_intervals=original.source_parameter_intervals,
        orientations=np.asarray(original.orientations),
        parent_dimensions=np.ones(ids.size, dtype=np.int8),
        parent_ids=ids + 1000,
        provenance=GeometryAssociationProvenance.LINEAGE,
        parent_association_id="original-curve-bank",
    )
    lineage, stencil = _identity_lineage(source.mesh)
    result = _curve_edges_with_root_ancestry(
        source, lineage, source.mesh, stencil, parent
    )
    assert result.parent_association_id == "original-curve-bank"
    # Replay rows follow the source curves, so each retained edge is compared
    # by its target id rather than by row position.
    replayed = np.asarray(result.target_global_ids, dtype=np.int64).tolist()
    assert sorted(replayed) == ids.tolist()
    expected = {
        int(target): (int(target) + 1000, int(curve), interval)
        for target, curve, interval in zip(
            ids.tolist(),
            np.asarray(parent.source_indices).tolist(),
            parent.source_parameter_intervals,
            strict=True,
        )
    }
    assert {
        target: (int(root), int(curve), interval)
        for target, root, curve, interval in zip(
            replayed,
            np.asarray(result.parent_ids, dtype=np.int64).tolist(),
            np.asarray(result.source_indices).tolist(),
            result.source_parameter_intervals,
            strict=True,
        )
    } == expected


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

    wrong_occurrence = GeometryAssociation(
        GeometryAssociationKind.MAPPED_REFERENCE,
        domain.source_id,
        domain.source_revision,
        mesh.entity_set(3).entity_set_id,
        np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64),
        cell_association.source_entity_ids,
        np.zeros((mesh.entity_set(3).count,), dtype=np.float64),
        exact=True,
        source_dimensions=np.full((mesh.entity_set(3).count,), 3, dtype=np.int8),
        source_indices=np.asarray(cell_association.source_indices),
        source_roles=(GeometrySourceEntityRole.REGION,) * mesh.entity_set(3).count,
        source_occurrence_paths=(("wrong-occurrence",),) * mesh.entity_set(3).count,
    )
    wrong_result = certify_cell_mesh(
        mesh,
        domain.coordinate_contract,
        geometry=geometry,
        associations=(wrong_occurrence,),
    )
    with pytest.raises(ValueError, match="occurrence"):
        mapped_transfer.validate_source(wrong_result)

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
    result = transfer.propagate(
        composed_source,
        lineage,
        mesh,
        vertex_stencil=stencil,
        geometry_transition=None,
        target_fidelity_witness=None,
        target_scoped_fidelity=(),
        validated_fidelity=(),
    )
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


def _implicit_ball_volume(
    source_id: str,
) -> tuple[CellMeshingResult, NativeImplicitSource]:
    profile = _profile(source_id)
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
    return source, authority


def test_implicit_volume_boundary_remap_orients_triangles_outward() -> None:
    source, authority = _implicit_ball_volume("association-ball")
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


def _preserved_lineage(source: CellMesh, target: CellMesh, /) -> MeshLineage:
    """Lineage of every entity whose sorted vertex-ID key both meshes keep."""
    records = []
    for dimension in range(4):
        source_ids = np.asarray(source.entity_set(dimension).entity_ids, dtype=np.int64)
        target_ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
        if dimension == 3:
            rows = entity_rows(target, 3, source_ids)
        else:
            rows = key_rows(
                entity_keys(target, dimension), entity_keys(source, dimension)
            )
        kept = rows >= 0
        records.append(
            EntityLineage(
                dimension,
                source.entity_set(dimension).entity_set_id,
                target.entity_set(dimension).entity_set_id,
                source_ids[kept],
                target_ids[rows[kept]],
                np.full(
                    (int(np.count_nonzero(kept)),),
                    EntityLineageKind.PRESERVED,
                    dtype=np.int32,
                ),
            )
        )
    return MeshLineage(source.topology_id, target.topology_id, tuple(records))


def test_implicit_composite_created_vertices_use_certified_support() -> None:
    """Created vertices whose exact stencil spans no single source simplex.

    Implicit-only boundary moves, collapses and flips join vertices across the
    source facets of one implicit group, so a later metric split creates a
    vertex whose stencil spans no source simplex (the offset-ring torus metric
    epoch). A zero-set vertex then needs the validated target root-chart
    witness of an incident face; a region vertex takes its incident cells'
    source ancestry.
    """
    source, authority = _implicit_ball_volume("association-composite-ball")
    transfer = ImplicitAssociationTransfer(authority)
    mesh = source.mesh
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    on_boundary = np.asarray(boundary_mask(mesh, 0), dtype=np.bool_)
    source_edges = {tuple(row) for row in entity_keys(mesh, 1).tolist()}
    boundary_ids = vertex_ids[on_boundary].tolist()
    interior_ids = vertex_ids[~on_boundary].tolist()
    assert len(boundary_ids) > 3 and interior_ids
    surface_vertex = boundary_ids[0]
    first = boundary_ids[1]
    second = next(
        value
        for value in boundary_ids[2:]
        if (min(first, value), max(first, value)) not in source_edges
    )
    region_vertex = interior_ids[0]
    region_support = sorted(
        value for value in vertex_ids.tolist() if value != region_vertex
    )[:5]
    born = int(np.max(vertex_ids)) + 1
    renamed = {surface_vertex: born, region_vertex: born + 1}
    target_ids = np.asarray(
        [renamed.get(value, value) for value in vertex_ids.tolist()], dtype=np.int64
    )
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    cell_ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    target = CellMesh.from_tetrahedra(
        np.asarray(mesh.coordinates),
        cells,
        vertex_global_ids=target_ids,
        cell_global_ids=cell_ids,
        numeric_version=mesh.numeric_version,
    )
    lineage = _preserved_lineage(mesh, target)
    exact_rows: dict[int, tuple[tuple[int, Fraction], ...]] = {
        born: ((first, Fraction(1, 2)), (second, Fraction(1, 2))),
        born + 1: tuple(
            zip(
                region_support,
                (
                    Fraction(1, 2),
                    Fraction(1, 4),
                    Fraction(1, 8),
                    Fraction(1, 16),
                    Fraction(1, 16),
                ),
                strict=True,
            )
        ),
    }
    width = 5
    sources = np.full((target_ids.size, width), -1, dtype=np.int64)
    weights = np.zeros((target_ids.size, width), dtype=np.float64)
    valid = np.zeros((target_ids.size, width), dtype=np.bool_)
    exact: list[tuple[Fraction, ...]] = []
    for row, identifier in enumerate(target_ids.tolist()):
        entries = exact_rows.get(identifier, ((identifier, Fraction(1)),))
        sources[row, : len(entries)] = [value for value, _ in entries]
        weights[row, : len(entries)] = [float(weight) for _, weight in entries]
        valid[row, : len(entries)] = True
        exact.append(
            tuple(weight for _, weight in entries)
            + (Fraction(0),) * (width - len(entries))
        )
    stencil = VertexInterpolationStencil(
        mesh.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        target_ids,
        sources,
        weights,
        valid,
        exact_weights=tuple(exact),
    )
    geometry = CellGeometrySpec.affine(target)
    assert source.certification is not None
    query = source.certification.inputs.source
    source_witness = source.certification.inputs.fidelity_witness
    assert query is not None and isinstance(source_witness, SourceBoundaryChartWitness)
    # The target keeps every source face; renamed vertices only rename its key.
    face_keys = entity_keys(mesh, 2)
    renamed_keys = np.sort(
        np.asarray(
            [[renamed.get(value, value) for value in row] for row in face_keys.tolist()],
            dtype=np.int64,
        ),
        axis=1,
    )
    target_face_ids = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)[
        key_rows(entity_keys(target, 2), renamed_keys)
    ]
    witness = _retarget_implicit_witness(
        source_witness,
        target,
        geometry,
        np.asarray(source_witness.cell_global_ids, dtype=np.int64),
        entity_ids=target_face_ids[
            entity_rows(mesh, 2, np.asarray(source_witness.entity_global_ids))
        ],
    )
    validated = validate_source_fidelity_witness(target, geometry, query, witness)

    def propagate(
        target_witness: SourceBoundaryChartWitness | None,
        tokens: tuple[ValidatedSourceFidelityWitness, ...],
    ) -> tuple[GeometryAssociation, ...]:
        return transfer.propagate(
            source,
            lineage,
            target,
            vertex_stencil=stencil,
            geometry_transition=None,
            target_fidelity_witness=target_witness,
            target_scoped_fidelity=(),
            validated_fidelity=tokens,
        )

    # Without the validated root chart, the composite zero-set stencil has no
    # source-stratum support.
    for target_witness, tokens in ((None, ()), (witness, ())):
        with pytest.raises(AssociationPropagationError, match="root-chart witness"):
            propagate(target_witness, tokens)
    associations = propagate(witness, (validated,))
    vertices = next(
        value for value in associations if target_dimension(target, value) == 0
    )
    assert vertices.complete
    surface_row, region_row = vertices.target_rows(np.asarray((born, born + 1)))
    assert vertices.source_roles[surface_row] is GeometrySourceEntityRole.FACET
    assert int(np.asarray(vertices.parent_dimensions)[surface_row]) == -1
    assert vertices.source_roles[region_row] is GeometrySourceEntityRole.REGION
    region_target_row = int(np.flatnonzero(target_ids == born + 1)[0])
    region_cells = cell_ids[np.any(cells == region_target_row, axis=1)]
    assert int(np.asarray(vertices.parent_dimensions)[region_row]) == 3
    assert int(np.asarray(vertices.parent_ids)[region_row]) == int(np.min(region_cells))


def test_implicit_chart_restriction_replays_in_an_independent_cold_process(
    tmp_path: Path,
) -> None:
    worker = (
        Path(__file__).resolve().parents[1]
        / "interchange"
        / "_native_cad_archive_worker.py"
    )
    written = subprocess.run(
        [sys.executable, str(worker), "write-implicit-chart-restriction", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(worker),
            "read-implicit-chart-restriction",
            str(tmp_path / "implicit-chart-restriction.phx"),
            _worker_oracle(tmp_path, expected),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert json.loads(restored.stdout) == expected


@pytest.fixture(scope="module")
def accepted_native_layer_core() -> _NativeLayerCoreFixture:
    """Generate the original ridge with its explicitly approved conforming cap."""
    from phydrax.meshing._layer_core_association_transfer import (
        NativeLayerCoreAssociationTransfer,
    )
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _accepted_layer_columns,
        _archived_ridge_source,
        _periodic_layer_core_specification,
    )

    source, _, _ = _archived_ridge_source(cap_refinement="conforming")
    specification = _periodic_layer_core_specification(source)
    result = (
        NativeMeshingProvider(NativeMeshingOptions("layer_core"))
        .plan(source, specification, coordinate_contract=_CONTRACT)
        .execute()
    )
    assert result.certification is not None and result.certification.passed
    adapter = NativeLayerCoreAssociationTransfer(source, specification)
    return source, specification, result, adapter, _accepted_layer_columns(result)


def _replace_association_support(
    table: GeometryAssociation,
    /,
    *,
    pieces: tuple[SourceBoundaryTargetPieceRow | None, ...] | None = None,
    exact: bool | None = None,
) -> GeometryAssociation:
    return GeometryAssociation(
        table.association_kind,
        table.source_id,
        table.source_revision,
        table.target_entity_set_id,
        table.target_global_ids,
        table.source_entity_ids,
        table.residuals,
        resolved=table.resolved,
        ambiguous=table.ambiguous,
        exact=table.exact if exact is None else exact,
        source_dimensions=table.source_dimensions,
        source_indices=table.source_indices,
        source_roles=table.source_roles,
        source_occurrence_paths=table.source_occurrence_paths,
        source_region_sides=table.source_region_sides,
        source_parameter_intervals=table.source_parameter_intervals,
        source_parameter_points=table.source_parameter_points,
        target_parameter_intervals=table.target_parameter_intervals,
        target_reference_pieces=table.target_reference_pieces
        if pieces is None
        else pieces,
        source_boundary_witness_id=table.source_boundary_witness_id,
        parameters=table.parameters,
        orientations=table.orientations,
        parent_dimensions=table.parent_dimensions,
        parent_ids=table.parent_ids,
        parent_association_id=table.parent_association_id,
        provenance=table.provenance,
    )


def test_face_image_association_consumer_preserves_distinct_source_strata(
    tmp_path: Path,
) -> None:
    from phydrax.meshing._association import (
        GeometryAssociationArchiveRecipe,
        GeometryAssociationBankArchiveRecipe,
        source_boundary_image_association_piece_rows,
        source_boundary_image_face_orientations,
    )
    from tests.unit.meshing.test_native_layer_overlap import (
        _cross_strata_face_image_fixture,
    )

    mesh, geometry, query, witness = _cross_strata_face_image_fixture()
    groups = source_boundary_image_association_piece_rows(witness)
    orientations = source_boundary_image_face_orientations(mesh, geometry, witness)
    tables: list[GeometryAssociation] = []
    for (face, identity, path), pieces in groups.items():
        row = next(
            index
            for index, (target, entity, occurrence) in enumerate(
                zip(
                    witness.entity_global_ids,
                    witness.source_entity_ids,
                    witness.occurrence_paths,
                    strict=True,
                )
            )
            if (int(target), entity, occurrence) == (face, identity, path)
        )
        table = GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            query.source_id,
            query.source_revision,
            mesh.entity_set(2).entity_set_id,
            np.asarray([face]),
            (identity,),
            np.zeros(1),
            exact=True,
            source_roles=(GeometrySourceEntityRole.FACET,),
            source_occurrence_paths=(path,),
            source_region_sides=(witness.source_region_sides[row],),
            orientations=np.asarray(
                [orientations[(face, identity, path)]], dtype=np.int8
            ),
            target_reference_pieces=(pieces,),
            source_boundary_witness_id=witness.witness_id,
        )
        table.validate_source_boundary_support(mesh, geometry, query, witness)
        restored = GeometryAssociationArchiveRecipe.from_association(table).restore()
        restored.validate_source_boundary_support(mesh, geometry, query, witness)
        assert restored.association_id == table.association_id
        tables.append(table)
    assert len({table.source_entity_ids[0] for table in tables}) == 2
    assert len({int(table.target_global_ids[0]) for table in tables}) == 1
    bank = GeometryAssociationBankArchiveRecipe.from_associations(tuple(tables))
    restored_bank = bank.restore()
    assert tuple(value.association_id for value in restored_bank) == tuple(
        value.association_id for value in tables
    )
    for table in restored_bank:
        table.validate_source_boundary_support(mesh, geometry, query, witness)
    with pytest.raises(ValueError):
        bank._replace(ends=np.asarray((0, int(bank.ends[-1])), dtype=np.int64)).restore()
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )

    entry = CompositionEntry(
        tuple(tables),
        entry_id="associations",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=mesh.topology_id,
        revision_id=mesh.mesh_id,
        semantics_id="source-associations",
    )
    path = tmp_path / "association-bank.phx"
    write_model_artifact(
        path,
        Composition((entry,), boundary_id="association-bank"),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    decoded = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    ).value("associations")
    assert tuple(value.association_id for value in decoded) == tuple(
        value.association_id for value in tables
    )
    for table in decoded:
        table.validate_source_boundary_support(mesh, geometry, query, witness)
    forged = eqx.tree_at(
        lambda value: value.orientations, tables[0], -tables[0].orientations
    )
    with pytest.raises(ValueError):
        forged.validate_source_boundary_support(mesh, geometry, query, witness)


def _layer_association_semantics(
    result: CellMeshingResult,
) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for table in result.associations:
        dimension = target_dimension(result.mesh, table)
        for row, identifier in enumerate(
            np.asarray(table.target_global_ids, dtype=np.int64)
        ):
            rows.append(
                (
                    table.source_id,
                    table.source_revision,
                    table.association_kind,
                    dimension,
                    int(identifier),
                    table.source_entity_ids[row],
                    table.source_roles[row],
                    table.source_occurrence_paths[row],
                    table.source_region_sides[row],
                    table.source_parameter_intervals[row],
                    table.source_parameter_points[row],
                    table.target_parameter_intervals[row],
                    table.target_reference_pieces[row],
                    table.source_boundary_witness_id,
                    tuple(
                        None if np.isnan(value) else float(value)
                        for value in np.asarray(table.parameters)[row]
                    ),
                    int(np.asarray(table.source_dimensions)[row]),
                    int(np.asarray(table.source_indices)[row]),
                    int(np.asarray(table.orientations)[row]),
                    float(np.asarray(table.residuals)[row]),
                    bool(np.asarray(table.resolved)[row]),
                    bool(np.asarray(table.ambiguous)[row]),
                    table.exact,
                    int(np.asarray(table.parent_dimensions)[row]),
                    int(np.asarray(table.parent_ids)[row]),
                    table.provenance,
                    table.parent_association_id,
                )
            )
    return tuple(sorted(rows, key=repr))


def test_original_layer_rehashed_carrier_forgery_is_not_authority(
    accepted_native_layer_core: _NativeLayerCoreFixture,
    tmp_path: Path,
) -> None:
    from copy import copy

    from phydrax._fingerprint import canonical_fingerprint
    from phydrax._model._structure import LayerCoreAcceptedResultArchiveRecipe
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        layer_core_authoring_recipe,
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_layer_core_accepted_result,
        restore_layer_core_authoring,
        restore_model_archive_value,
        write_model_artifact,
    )

    source, specification, result, _, _ = accepted_native_layer_core
    assert source.layer_control is not None and source.layer_policy is not None
    carrier = copy(result.layer_core_carrier_witness)
    assert carrier is not None and carrier.original_source_edge_rows
    forged_rows = carrier.original_source_edge_rows[:-1]
    object.__setattr__(carrier, "original_source_edge_rows", forged_rows)
    object.__setattr__(
        carrier,
        "witness_id",
        canonical_fingerprint(
            {
                "kind": "native-layer-core-carrier-witness",
                "source": carrier.source_binding_id,
                "specification": carrier.specification_id,
                "original_source_edge_rows": forged_rows,
                "required_source_edge_rows": carrier.required_source_edge_rows,
            }
        ),
    )
    forged_result = copy(result)
    object.__setattr__(forged_result, "layer_core_carrier_witness", carrier)
    source_entry = CompositionEntry(
        layer_core_authoring_recipe(source, source.layer_control, source.layer_policy),
        entry_id="source",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=source.layers.mesh.topology_id,
        revision_id=source.binding_id,
        semantics_id="original-layer-core-authority",
    )
    spec_entry = CompositionEntry(
        specification,
        entry_id="specification",
        role="model-state",
        owner_id="phydrax.meshing",
        structure_id=source.binding_id,
        revision_id=specification.specification_id,
        semantics_id="original-layer-core-request",
        dependencies=(source_entry.binding("revision"),),
    )
    result_entry = CompositionEntry(
        LayerCoreAcceptedResultArchiveRecipe(
            forged_result, source.binding_id, specification.specification_id
        ),
        entry_id="result",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=forged_result.mesh.topology_id,
        revision_id=forged_result.result_id,
        semantics_id="untrusted-forged-carrier",
        dependencies=(source_entry.binding("revision"), spec_entry.binding("revision")),
    )
    path = tmp_path / "forged-carrier.phx"
    write_model_artifact(
        path,
        Composition(
            (source_entry, spec_entry, result_entry), boundary_id="carrier-forgery"
        ),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    receipt = model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS)
    archive = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        expected_content_id=receipt,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    restored_specification = VolumeMeshingSpec.from_stored_fields(
        archive.value("specification")
    )
    restored_source = restore_layer_core_authoring(
        archive.value("source"),
        specification=restored_specification,
    )
    with pytest.raises(ValueError, match="Stored native carrier inventory changed"):
        restore_layer_core_accepted_result(
            archive.value("result"),
            restored_source,
            restored_specification,
        )


def test_native_layer_core_mixed_round_trip_preserves_source_strata(
    accepted_native_layer_core: _NativeLayerCoreFixture,
    tmp_path: Path,
) -> None:
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.meshing._layer_core_subdivision import (
        require_refined_column_certification,
    )
    from phydrax.meshing._layer_source_association_replay import (
        replay_layer_source_associations,
    )
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _periodic_mixed_coarsening,
        _periodic_mixed_refinement,
    )

    source, specification, original, adapter, columns = accepted_native_layer_core
    import pickle
    from copy import copy

    from phydrax.meshing._layer_core_root_geometry import ComputedLayerCoreRoot

    root = original.layer_core_root_geometry
    assert root is not None
    computed_root = ComputedLayerCoreRoot.reprove(source, specification, root)
    computed_root.require(source, specification, root.mesh, root.geometry, root)
    for operands in (
        (copy(source), specification, root.mesh, root.geometry, root),
        (source, copy(specification), root.mesh, root.geometry, root),
        (source, specification, copy(root.mesh), root.geometry, root),
        (source, specification, root.mesh, root.geometry, copy(root)),
    ):
        with pytest.raises(ValueError, match="different exact operands"):
            computed_root.require(*operands)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(computed_root)
    from phydrax.discretization._nested_reference_types import ReferenceCellPartition
    from phydrax.meshing._model_codec import restore_layer_core_root_geometry

    first = root.layer_root_partition.maps[0]
    assert len(first.target_cell_ids) > 1
    missing = first._replace(
        target_cell_ids=first.target_cell_ids[1:],
        source_cell_ids=first.source_cell_ids[1:],
        source_reference_vertices=first.source_reference_vertices[1:],
        source_reference_exact_vertices=(
            None
            if first.source_reference_exact_vertices is None
            else first.source_reference_exact_vertices[1:]
        ),
        source_reference_chart_arguments=(
            None
            if first.source_reference_chart_arguments is None
            else first.source_reference_chart_arguments[1:]
        ),
    )
    forged_root = copy(root)
    object.__setattr__(
        forged_root,
        "layer_root_partition",
        ReferenceCellPartition((missing, *root.layer_root_partition.maps[1:])),
    )
    with pytest.raises(ValueError):
        restore_layer_core_root_geometry(forged_root, source, specification)
    overlapping_root = copy(root)
    object.__setattr__(
        overlapping_root,
        "layer_root_partition",
        ReferenceCellPartition((*root.layer_root_partition.maps, first)),
    )
    with pytest.raises(ValueError):
        restore_layer_core_root_geometry(overlapping_root, source, specification)
    adapter.validate_source(original)
    _, refinement, policy = _periodic_mixed_refinement(
        original, columns, adapter, None, mark_layer=True
    )
    refined = refinement.target
    from phydrax.meshing._layer_source_association_replay import _conforming_current_rows

    retained_geometry = refined.geometry
    altered_coordinates = np.array(
        retained_geometry.coordinates, dtype=np.float64, copy=True
    )
    routed_row = int(np.asarray(retained_geometry.geometry_dofs[0])[0, 0])
    altered_coordinates[routed_row, 0] += 0.001
    altered_geometry = CellGeometrySpec(
        dict(zip(retained_geometry.block_names, retained_geometry.elements, strict=True)),
        dict(
            zip(
                retained_geometry.block_names,
                retained_geometry.geometry_dofs,
                strict=True,
            )
        ),
        altered_coordinates,
        restriction_source=retained_geometry.restriction_source,
    )
    with pytest.raises(ValueError):
        tuple(_conforming_current_rows(root, refined.mesh, altered_geometry))
    _, restoration = _periodic_mixed_coarsening(original, refinement, policy)
    restored = restoration.target
    adapter.validate_source(refined)
    adapter.validate_source(restored)
    root = original.layer_core_root_geometry
    assert root is not None
    for endpoint in (original, refined, restored):
        assert endpoint.layer_core_root_geometry is not None
        assert endpoint.layer_core_root_geometry.root_id == root.root_id
        assert endpoint.layer_core_column_certification is not None
        require_refined_column_certification(
            source,
            endpoint.mesh,
            endpoint.geometry,
            specification,
            endpoint.layer_core_column_certification,
        )
        assert endpoint.certification is not None
        inputs = endpoint.certification.inputs
        assert inputs.cell_regions is not None
        actual_regions = dict(
            zip(
                np.asarray(inputs.cell_global_ids, dtype=np.int64).tolist(),
                np.asarray(inputs.cell_regions, dtype=np.int64).tolist(),
                strict=True,
            )
        )
        replay = replay_layer_source_associations(
            source, endpoint.mesh, endpoint.geometry, root
        )
        assert replay.layer_cell_ids and replay.core_cell_ids
        layer_materials = set(np.asarray(source.layer_regions, dtype=np.int64).tolist())
        core_materials = set(np.asarray(source.core_region_map, dtype=np.int64).tolist())
        assert all(
            actual_regions[cell] in layer_materials for cell in replay.layer_cell_ids
        )
        assert all(
            actual_regions[cell] in core_materials for cell in replay.core_cell_ids
        )
    for endpoint in (refined, restored):
        restriction = endpoint.geometry.restriction_source
        assert restriction is not None
        assert restriction.source_topology_id == root.mesh.topology_id
        assert restriction.source_geometry_id == cell_geometry_id(root.geometry)

    original_kinds = {
        (table.source_id, table.source_revision, table.association_kind)
        for table in original.associations
    }
    assert original_kinds <= {
        (table.source_id, table.source_revision, table.association_kind)
        for table in refined.associations
    }
    assert {
        (table.source_id, table.source_revision, table.association_kind)
        for table in restored.associations
    } == original_kinds
    assert any(
        target_dimension(refined.mesh, table) == 1
        and any(role is GeometrySourceEntityRole.FACET for role in table.source_roles)
        for table in refined.associations
    )
    assert any(
        interval is not None and interval[0] < interval[1]
        for table in refined.associations
        for interval in table.source_parameter_intervals
    )
    assert any(
        int(orientation) in (-1, 1)
        for table in refined.associations
        for orientation in np.asarray(table.orientations)
    )
    assert any(
        all(value is not None for value in sides)
        for table in refined.associations
        for sides in table.source_region_sides
    )
    assert _layer_association_semantics(restored) == _layer_association_semantics(
        original
    )
    from phydrax.meshing._layer_core_association import _core_source_identity

    authored = source if source.reference_source is None else source.reference_source
    for endpoint in (original, refined, restored):
        for edge, path in zip(
            authored.complex.plc_edge_ids,
            authored.complex.plc_edge_occurrence_paths,
            strict=True,
        ):
            identity, occurrence = _core_source_identity(
                authored, 1, str(edge), tuple(path)
            )
            intervals = sorted(
                interval
                for table in endpoint.associations
                if (table.source_id, table.source_revision)
                == (authored.source_id, authored.source_revision)
                and target_dimension(endpoint.mesh, table) in (1, 2)
                for row, interval in enumerate(table.source_parameter_intervals)
                if table.source_roles[row] is GeometrySourceEntityRole.EDGE
                and table.source_entity_ids[row] == identity
                and table.source_occurrence_paths[row] == occurrence
                and interval is not None
            )
            cursor = Fraction(0)
            for lower, upper in intervals:
                assert lower == cursor and upper > lower
                cursor = upper
            assert cursor == Fraction(1)
    assert any(
        role is GeometrySourceEntityRole.FACET
        and pieces is not None
        and any(isinstance(piece, SourceBoundaryImagePullback) for piece in pieces)
        for endpoint in (original, refined, restored)
        for table in endpoint.associations
        for role, pieces in zip(
            table.source_roles, table.target_reference_pieces, strict=True
        )
    )

    overlay_endpoint, overlay_index, overlay_row = next(
        (
            (endpoint, index, row)
            for endpoint in (refined, original)
            for index, table in enumerate(endpoint.associations)
            if target_dimension(endpoint.mesh, table) == 2
            for row, role in enumerate(table.source_roles)
            if role is GeometrySourceEntityRole.EDGE
            and (pieces := table.target_reference_pieces[row]) is not None
            and any(isinstance(piece, SourceBoundaryImagePullback) for piece in pieces)
        ),
        (None, None, None),
    )
    assert (
        overlay_endpoint is not None
        and overlay_index is not None
        and overlay_row is not None
    )
    overlay = overlay_endpoint.associations[overlay_index]
    face_ids = np.asarray(overlay_endpoint.mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_row = int(
        np.flatnonzero(face_ids == int(overlay.target_global_ids[overlay_row]))[0]
    )
    assert int(adapter.classes(overlay_endpoint)[2].dimensions[face_row]) != 1
    assert overlay_endpoint.certification is not None
    inputs = overlay_endpoint.certification.inputs
    query, witness = next(
        (query, witness)
        for query, witness in (
            (inputs.source, inputs.fidelity_witness),
            *((query, witness) for query, witness, _ in inputs.scoped_fidelity),
        )
        if isinstance(
            witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        )
        and witness.witness_id == overlay.source_boundary_witness_id
    )
    assert query is not None and isinstance(
        witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
    )
    overlay.validate_source_boundary_support(
        overlay_endpoint.mesh,
        overlay_endpoint.geometry,
        query,
        witness,
        association_source=source,
        layer_core_root_geometry=root,
    )
    with pytest.raises(ValueError):
        overlay.validate_source_boundary_support(
            overlay_endpoint.mesh,
            overlay_endpoint.geometry,
            query,
            witness,
        )
    row_pieces = overlay.target_reference_pieces[overlay_row]
    assert row_pieces is not None
    original_piece = row_pieces[0]
    if isinstance(original_piece, SourceBoundaryImagePullback):
        points = original_piece.image_rational_tokens
    else:
        _, _, _, points = original_piece
    assert len(points) == 2
    first_point = list(points[0])
    numerator, denominator = first_point[0]
    first_point[0] = (numerator + denominator, denominator)
    if tuple(first_point) == points[1]:
        first_point[0] = (numerator + 2 * denominator, denominator)
    if isinstance(original_piece, SourceBoundaryImagePullback):
        forged_piece = SourceBoundaryImagePullback(
            original_piece.cell_global_id,
            original_piece.local_entity_index,
            (tuple(first_point), points[1]),
            original_piece.source_chart_id,
        )
    else:
        cell, local, vertices, _ = original_piece
        forged_piece = (cell, local, vertices, (tuple(first_point), points[1]))
    altered_pieces = list(overlay.target_reference_pieces)
    altered_pieces[overlay_row] = tuple(
        sorted(
            (forged_piece, *row_pieces[1:]),
            key=_target_piece_sort_key,
        )
    )
    forged_overlay = _replace_association_support(overlay, pieces=tuple(altered_pieces))
    with pytest.raises(ValueError):
        forged_overlay.validate_source_boundary_support(
            overlay_endpoint.mesh,
            overlay_endpoint.geometry,
            query,
            witness,
            association_source=source,
            layer_core_root_geometry=root,
        )
    altered = list(overlay_endpoint.associations)
    altered[overlay_index] = forged_overlay
    forged_endpoint = eqx.tree_at(
        lambda value: value.associations, overlay_endpoint, tuple(altered)
    )
    with pytest.raises(ValueError):
        adapter.validate_source(forged_endpoint)
    omitted = tuple(
        table
        for index, table in enumerate(overlay_endpoint.associations)
        if index != overlay_index
    )
    missing_endpoint = eqx.tree_at(
        lambda value: value.associations, overlay_endpoint, omitted
    )
    with pytest.raises(ValueError):
        adapter.validate_source(missing_endpoint)
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        layer_core_accepted_result_recipe,
        layer_core_authoring_recipe,
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        write_model_artifact,
    )

    assert source.layer_control is not None and source.layer_policy is not None
    source_entry = CompositionEntry(
        layer_core_authoring_recipe(source, source.layer_control, source.layer_policy),
        entry_id="source",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=source.layers.mesh.topology_id,
        revision_id=source.binding_id,
        semantics_id="original-layer-core-authority",
    )
    specification_entry = CompositionEntry(
        specification,
        entry_id="specification",
        role="model-state",
        owner_id="phydrax.meshing",
        structure_id=source.binding_id,
        revision_id=specification.specification_id,
        semantics_id="original-layer-core-request",
        dependencies=(source_entry.binding("revision"),),
    )
    result_entry = CompositionEntry(
        layer_core_accepted_result_recipe(overlay_endpoint, source, specification),
        entry_id="result",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=overlay_endpoint.mesh.topology_id,
        revision_id=overlay_endpoint.result_id,
        semantics_id="accepted-image-support",
        dependencies=(
            source_entry.binding("revision"),
            specification_entry.binding("revision"),
        ),
    )
    archive_path = tmp_path / "layer-core-image.npz"
    write_model_artifact(
        archive_path,
        Composition(
            (source_entry, specification_entry, result_entry),
            boundary_id="layer-core-image-cold-proof",
        ),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    receipt = model_archive_content_id(archive_path, limits=MODEL_ARCHIVE_LIMITS)
    # A content hash can authenticate a forged recipe but cannot establish
    # source inventory. Rehash both the false witness and its whole archive.
    from phydrax._fingerprint import canonical_fingerprint
    from phydrax._model._structure import LayerCoreAcceptedResultArchiveRecipe
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_layer_core_accepted_result,
        restore_model_archive_value,
    )

    carrier = copy(overlay_endpoint.layer_core_carrier_witness)
    assert carrier is not None and carrier.original_source_edge_rows
    forged_rows = carrier.original_source_edge_rows[:-1]
    object.__setattr__(carrier, "original_source_edge_rows", forged_rows)
    object.__setattr__(
        carrier,
        "witness_id",
        canonical_fingerprint(
            {
                "kind": "native-layer-core-carrier-witness",
                "source": carrier.source_binding_id,
                "specification": carrier.specification_id,
                "original_source_edge_rows": forged_rows,
                "required_source_edge_rows": carrier.required_source_edge_rows,
            }
        ),
    )
    forged_result = copy(overlay_endpoint)
    object.__setattr__(forged_result, "layer_core_carrier_witness", carrier)
    forged_entry = CompositionEntry(
        LayerCoreAcceptedResultArchiveRecipe(
            forged_result, source.binding_id, specification.specification_id
        ),
        entry_id="result",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=forged_result.mesh.topology_id,
        revision_id=forged_result.result_id,
        semantics_id="accepted-image-support",
        dependencies=(
            source_entry.binding("revision"),
            specification_entry.binding("revision"),
        ),
    )
    forged_path = tmp_path / "layer-core-forged.npz"
    write_model_artifact(
        forged_path,
        Composition(
            (source_entry, specification_entry, forged_entry),
            boundary_id="layer-core-image-cold-proof",
        ),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    forged_receipt = model_archive_content_id(forged_path, limits=MODEL_ARCHIVE_LIMITS)
    forged_archive = read_model_artifact(
        forged_path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
        expected_content_id=forged_receipt,
    )
    with pytest.raises(ValueError, match="Stored native carrier inventory changed"):
        restore_layer_core_accepted_result(
            forged_archive.value("result"), source, specification
        )
    child = """
import json
import sys
import equinox as eqx
from phydrax.lifecycle import Composition
from phydrax.meshing._association import GeometryAssociation, GeometrySourceEntityRole
from phydrax.geometry._source_boundary_records import SourceBoundaryFaceImageWitness, SourceBoundaryImagePullback
from phydrax.meshing._contracts import VolumeMeshingSpec
from phydrax.meshing._layer_core_association_transfer import NativeLayerCoreAssociationTransfer
from phydrax.meshing._model_codec import (
    MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_layer_core_accepted_result,
    restore_layer_core_authoring, restore_model_archive_value,
)

archive = read_model_artifact(
    sys.argv[1], expected_type=Composition, restore=restore_model_archive_value,
    limits=MODEL_ARCHIVE_LIMITS, expected_content_id=sys.argv[2],
)
specification = VolumeMeshingSpec.from_stored_fields(archive.value("specification"))
source = restore_layer_core_authoring(archive.value("source"), specification=specification)
result = restore_layer_core_accepted_result(archive.value("result"), source, specification)
adapter = NativeLayerCoreAssociationTransfer(source, specification)
assert result.certification is not None and result.layer_core_root_geometry is not None
inputs = result.certification.inputs
sources = ((inputs.source, inputs.fidelity_witness),) + tuple(
    (query, witness) for query, witness, _ in inputs.scoped_fidelity
)
images = []
for index, table in enumerate(result.associations):
    if not any(
        pieces is not None and any(isinstance(piece, SourceBoundaryImagePullback) for piece in pieces)
        for pieces in table.target_reference_pieces
    ):
        continue
    query, witness = next(
        (query, witness) for query, witness in sources
        if query is not None and isinstance(witness, SourceBoundaryFaceImageWitness)
        and witness.witness_id == table.source_boundary_witness_id
    )
    images.append((index, table, query, witness))
facet = next(item for item in images if any(
    role is GeometrySourceEntityRole.FACET and a is not None and b is not None and a != b
    for role, (a, b) in zip(item[1].source_roles, item[1].source_region_sides, strict=True)
))
edge = next(item for item in images if any(
    role is GeometrySourceEntityRole.EDGE for role in item[1].source_roles
))
for _, table, query, witness in (facet, edge):
    table.validate_source_boundary_support(
        result.mesh, result.geometry, query, witness,
        association_source=source, layer_core_root_geometry=result.layer_core_root_geometry,
    )

def rebuilt(table, *, sides=None, witness_id=None):
    return GeometryAssociation(
        table.association_kind, table.source_id, table.source_revision,
        table.target_entity_set_id, table.target_global_ids, table.source_entity_ids,
        table.residuals, resolved=table.resolved, ambiguous=table.ambiguous, exact=table.exact,
        source_dimensions=table.source_dimensions, source_indices=table.source_indices,
        source_roles=table.source_roles, source_occurrence_paths=table.source_occurrence_paths,
        source_region_sides=table.source_region_sides if sides is None else sides,
        source_parameter_intervals=table.source_parameter_intervals,
        source_parameter_points=table.source_parameter_points,
        target_parameter_intervals=table.target_parameter_intervals,
        target_reference_pieces=table.target_reference_pieces,
        source_boundary_witness_id=table.source_boundary_witness_id if witness_id is None else witness_id,
        parameters=table.parameters, orientations=table.orientations,
        parent_dimensions=table.parent_dimensions, parent_ids=table.parent_ids,
        parent_association_id=table.parent_association_id, provenance=table.provenance,
    )

_, table, query, witness = facet
sides = list(table.source_region_sides)
row = next(i for i, (a, b) in enumerate(sides) if a is not None and b is not None and a != b)
sides[row] = tuple(reversed(sides[row]))
for forged in (rebuilt(table, sides=tuple(sides)), rebuilt(table, witness_id="forged-image-witness")):
    try:
        forged.validate_source_boundary_support(result.mesh, result.geometry, query, witness,
            association_source=source, layer_core_root_geometry=result.layer_core_root_geometry)
    except ValueError:
        pass
    else:
        raise AssertionError("Cold image support accepted forged sides or witness identity.")
remaining = tuple(table for index, table in enumerate(result.associations) if index != edge[0])
missing = eqx.tree_at(lambda value: value.associations, result, remaining)
try:
    adapter.validate_source(missing)
except ValueError:
    pass
else:
    raise AssertionError("Cold image source accepted an omitted source-edge partition.")
print(json.dumps({"source": source.binding_id, "result": result.result_id}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", child, str(archive_path), receipt],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    restored_ids = json.loads(completed.stdout.strip().splitlines()[-1])
    assert restored_ids == {
        "source": source.binding_id,
        "result": overlay_endpoint.result_id,
    }


def test_native_layer_core_refuses_stale_forged_and_conflicting_sources(
    accepted_native_layer_core: _NativeLayerCoreFixture,
) -> None:
    from phydrax.meshing._layer_core_association_transfer import (
        NativeLayerCoreAssociationTransfer,
    )

    source, specification, result, adapter, columns = accepted_native_layer_core
    changed_regions = np.asarray(source.layer_regions, dtype=np.int64).copy()
    changed_regions[0] = len(source.region_ids)
    stale = eqx.tree_at(lambda value: value.layer_regions, source, changed_regions)
    with pytest.raises(ValueError):
        NativeLayerCoreAssociationTransfer(stale, specification)

    table_index = next(
        index
        for index, table in enumerate(result.associations)
        if target_dimension(result.mesh, table) == 2
        and table.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
        and table.exact
        and np.any(np.asarray(table.orientations) != 0)
        and any(
            all(value is not None for value in sides)
            for sides in table.source_region_sides
        )
    )
    original = result.associations[table_index]
    forged = eqx.tree_at(
        lambda value: value.orientations, original, -original.orientations
    )
    altered = list(result.associations)
    altered[table_index] = forged
    forged_result = eqx.tree_at(lambda value: value.associations, result, tuple(altered))
    with pytest.raises(ValueError):
        adapter.validate_source(forged_result)
    request = MarkedMeshAdaptation(
        np.asarray((int(columns.cell_ids[0]),), dtype=np.int64),
        layer_columns=columns,
    )
    with pytest.raises(ValueError):
        prepare_mesh_adaptation(
            forged_result,
            request,
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_MIXED, association_transfer=adapter
            ),
        )
    forged_exactness = _replace_association_support(original, exact=False)
    altered[table_index] = forged_exactness
    forged_result = eqx.tree_at(lambda value: value.associations, result, tuple(altered))
    with pytest.raises(ValueError):
        adapter.validate_source(forged_result)

    sides_index = next(
        index
        for index, sides in enumerate(original.source_region_sides)
        if all(value is not None for value in sides)
    )
    sides = list(original.source_region_sides)
    sides[sides_index] = (sides[sides_index][1], sides[sides_index][0])
    conflicting = GeometryAssociation(
        original.association_kind,
        original.source_id,
        original.source_revision,
        original.target_entity_set_id,
        original.target_global_ids,
        original.source_entity_ids,
        original.residuals,
        resolved=original.resolved,
        ambiguous=original.ambiguous,
        exact=original.exact,
        source_roles=original.source_roles,
        source_occurrence_paths=original.source_occurrence_paths,
        source_region_sides=tuple(sides),
        source_parameter_intervals=original.source_parameter_intervals,
        source_parameter_points=original.source_parameter_points,
        target_parameter_intervals=original.target_parameter_intervals,
        target_reference_pieces=original.target_reference_pieces,
        source_boundary_witness_id=original.source_boundary_witness_id,
        parameters=original.parameters,
        orientations=original.orientations,
    )
    conflicting_result = eqx.tree_at(
        lambda value: value.associations, result, (*result.associations, conflicting)
    )
    with pytest.raises(ValueError):
        adapter.validate_source(conflicting_result)


@pytest.mark.parametrize(
    "forgery", ("missing-child", "wrong-parent", "wrong-target", "swapped-target")
)
def test_brep_volume_curve_association_survives_refinement_and_rejects_stale(
    forgery: str,
) -> None:
    from copy import copy

    import phydrax as phx
    from phydrax.geometry.brep import brep_box
    from phydrax.meshing._sizing import UniformSizeControl

    model = brep_box(
        np.zeros(3),
        np.ones(3),
        coordinate_contract=_CONTRACT,
        source_id="test:brep-volume-curve-transfer",
    )
    boundary = MeshingScope.whole(model, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, 0.5),),
        maximum_deviation=0.02,
    )
    source = (
        NativeMeshingProvider()
        .plan(model, specification, coordinate_contract=_CONTRACT)
        .execute()
    )
    assert source.certification is not None
    assert isinstance(
        source.certification.inputs.fidelity_witness, SourceBoundaryChartWitness
    )

    def witness_of(value: phx.meshing.CellMeshingResult) -> SourceBoundaryChartWitness:
        assert value.certification is not None
        witness = value.certification.inputs.fidelity_witness
        assert isinstance(witness, SourceBoundaryChartWitness)
        return witness

    def associations_and_witness(
        value: phx.meshing.CellMeshingResult,
    ) -> tuple[tuple[GeometryAssociation, ...], SourceBoundaryChartWitness]:
        return value.associations, witness_of(value)

    def curve_and_witness(
        value: phx.meshing.CellMeshingResult,
    ) -> tuple[GeometryAssociation, SourceBoundaryChartWitness]:
        return value.associations[curve_index], witness_of(value)

    transfer = phx.meshing.BRepAssociationTransfer(
        phx.geometry.prepare_brep_projection(model)
    )
    curve = next(
        item
        for item in source.associations
        if item.association_kind is GeometryAssociationKind.CURVE
    )
    transfer.validate_source(source)
    foreign = brep_box(
        np.zeros(3),
        np.ones(3),
        coordinate_contract=_CONTRACT,
        source_id="test:foreign-volume-curve-transfer",
    )
    with pytest.raises(ValueError, match="vertex association"):
        phx.meshing.BRepAssociationTransfer(
            phx.geometry.prepare_brep_projection(foreign)
        ).validate_source(source)
    stale = eqx.tree_at(
        witness_of,
        source,
        None,
    )
    with pytest.raises(ValueError, match="source chart witness"):
        transfer.validate_source(stale)
    curve_index = next(i for i, item in enumerate(source.associations) if item is curve)
    invalid_curve = copy(curve)
    object.__setattr__(invalid_curve, "association_kind", GeometryAssociationKind.SURFACE)
    invalid = eqx.tree_at(
        lambda value: value.associations[curve_index], source, invalid_curve
    )
    with pytest.raises(ValueError, match="mix association kinds"):
        transfer.validate_source(invalid)
    absent_witness = copy(source.certification.inputs.fidelity_witness)
    object.__setattr__(absent_witness, "cad_curve_association", None)
    without_curve = eqx.tree_at(
        associations_and_witness,
        source,
        (
            source.associations[:curve_index] + source.associations[curve_index + 1 :],
            absent_witness,
        ),
    )
    with pytest.raises(ValueError, match="curve association is missing"):
        transfer.validate_source(without_curve)
    rows = np.arange(len(curve.target_global_ids), dtype=np.int64)
    if forgery == "missing-child":
        rows = rows[1:]
    parent_ids = np.asarray(curve.parent_ids)[rows].copy()
    if forgery == "wrong-parent":
        parent_ids[0] = next(value for value in parent_ids if value != parent_ids[0])
    target_ids = np.asarray(curve.target_global_ids)[rows].copy()
    if forgery == "wrong-target":
        target_ids[0] = next(
            identifier
            for identifier in np.asarray(source.mesh.entity_set(1).entity_ids)
            if identifier not in target_ids
        )
    if forgery == "swapped-target":
        first, second = next(
            (i, j)
            for i in range(len(target_ids))
            for j in range(i + 1, len(target_ids))
            if curve.source_entity_ids[i] == curve.source_entity_ids[j]
            and curve.source_occurrence_paths[i] == curve.source_occurrence_paths[j]
        )
        target_ids[first], target_ids[second] = target_ids[second], target_ids[first]

    def selected[T](values: tuple[T, ...]) -> tuple[T, ...]:
        return tuple(values[int(row)] for row in rows)

    forged_curve = GeometryAssociation(
        curve.association_kind,
        curve.source_id,
        curve.source_revision,
        curve.target_entity_set_id,
        target_ids,
        selected(curve.source_entity_ids),
        np.asarray(curve.residuals)[rows],
        resolved=np.asarray(curve.resolved)[rows],
        ambiguous=np.asarray(curve.ambiguous)[rows],
        exact=curve.exact,
        source_dimensions=np.asarray(curve.source_dimensions)[rows],
        source_indices=np.asarray(curve.source_indices)[rows],
        source_roles=selected(curve.source_roles),
        source_occurrence_paths=selected(curve.source_occurrence_paths),
        source_region_sides=selected(curve.source_region_sides),
        source_parameter_intervals=selected(curve.source_parameter_intervals),
        target_parameter_intervals=selected(curve.target_parameter_intervals),
        target_reference_pieces=selected(curve.target_reference_pieces),
        source_boundary_witness_id=curve.source_boundary_witness_id,
        source_parameter_points=selected(curve.source_parameter_points),
        parameters=np.asarray(curve.parameters)[rows],
        orientations=np.asarray(curve.orientations)[rows],
        parent_dimensions=np.asarray(curve.parent_dimensions)[rows],
        parent_ids=parent_ids,
        parent_association_id=curve.parent_association_id,
        provenance=curve.provenance,
    )
    forged_witness = copy(source.certification.inputs.fidelity_witness)
    object.__setattr__(forged_witness, "cad_curve_association", forged_curve)
    forged_source = eqx.tree_at(
        curve_and_witness,
        source,
        (forged_curve, forged_witness),
    )
    with pytest.raises(ValueError):
        transfer.validate_source(forged_source)
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(np.asarray(source.mesh.blocks[0].global_ids[:1])),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION, association_transfer=transfer
            ),
        )
    )
    target = result.target
    target_curve = next(
        item
        for item in target.associations
        if item.association_kind is GeometryAssociationKind.CURVE
    )
    assert target.certification is not None
    target_witness = target.certification.inputs.fidelity_witness
    assert isinstance(target_witness, SourceBoundaryChartWitness)
    assert target_witness.cad_curve_association is not None
    assert target_curve.parent_association_id == curve.parent_association_id
    assert target_curve.association_id == (
        target_witness.cad_curve_association.association_id
    )
    assert all(
        interval is not None for interval in target_curve.source_parameter_intervals
    )
    assert source.surface_source is not None
    root = next(
        item
        for item in source.surface_source.associations
        if item.association_id == curve.parent_association_id
    )
    expected = dict(
        zip(
            np.asarray(root.target_global_ids).tolist(),
            root.source_parameter_intervals,
            strict=True,
        )
    )
    assert all(interval is not None for interval in root.source_parameter_intervals)
    for current in (curve, target_curve):
        grouped = {}
        for parent, interval in zip(
            np.asarray(current.parent_ids).tolist(),
            current.source_parameter_intervals,
            strict=True,
        ):
            assert interval is not None
            grouped.setdefault(parent, []).append(interval)
        assert set(grouped) == set(expected)
        for parent, values in grouped.items():
            expected_interval = expected[parent]
            assert expected_interval is not None
            intervals = sorted(values)
            assert intervals[0][0] == expected_interval[0]
            assert intervals[-1][1] == expected_interval[1]
            assert all(
                left[1] == right[0] for left, right in zip(intervals, intervals[1:])
            )
    transfer.validate_source(target)

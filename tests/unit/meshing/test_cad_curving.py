"""CAD association transitions, continuous-fidelity refusals and node budgets."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.meshing._curving import HighOrderCurvingResult, straight_geometry
from phydrax.meshing._surface_boundary_sources import SurfaceSubdivisionBoundarySource
from tests._support.cad_curving import (
    association_classes,
    collapsed_mesh,
    coordinate_rows,
    CURVING_ASSOCIATION_POLICY,
    CURVING_COORDINATE_CONTRACT,
    CURVING_STATUS,
    icosphere,
    plate_mesh,
    plate_projection,
    sphere_error,
    sphere_projection,
)


# Unchanged empirical allowance from the pre-refactor straight-edge, seam, plane-distance and zero-mean reference assertions.
CURVING_COORDINATE_ROUNDOFF_ATOL = 1e-12
# Unchanged empirical allowance from the pre-refactor independent plate interior/boundary classification assertions.
BOUNDARY_CLASSIFICATION_MARGIN = 1e-9


@pytest.fixture(scope="module")
def native_cad_subdivision() -> tuple[
    phx.meshing.CellMeshingResult, phx.meshing.CellMeshingResult, phx.meshing.PreparedMeshAdaptation
]:
    from tools.meshing_qualification._controls import authored_resource_envelope

    envelope = authored_resource_envelope(
        "native-surface-pde", "planar-hole-feature", {}, larger=False
    )
    assert envelope is not None
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(((0, 0), (1, 0), (1, 1), (0, 1))),
    )
    model = phx.geometry.brep_planar_face(
        profile, coordinate_contract=CURVING_COORDINATE_CONTRACT
    )
    domain = phx.geometry.MeshingDomain.from_brep(model)
    scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        # The plane's zero curvature leaves the hard edge bound in control.
        size_controls=(
            phx.meshing.CurvatureSizeControl(
                scope,
                0.1,
                maximum_size=0.5,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        # This planar source uses the registered whole-workflow envelope.
        limits=phx.meshing.MeshingLimits(
            maximum_work_units=envelope["effective"]["maximum_work_units"],
            maximum_geometry_queries=envelope["effective"]["maximum_geometry_queries"],
            maximum_scratch_bytes=envelope["effective"]["maximum_scratch_bytes"],
        ),
    )
    original = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("parametric_surface")
        )
        .plan(
            phx.meshing.NativeSurfaceSource(domain),
            specification,
            coordinate_contract=CURVING_COORDINATE_CONTRACT,
        )
        .execute()
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        limits=specification.limits,
        compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
        association_transfer=phx.meshing.BRepAssociationTransfer(
            phx.geometry.prepare_brep_projection(model)
        ),
    )
    current = original
    hierarchy = None
    for _ in range(2):
        prepared = phx.meshing.prepare_mesh_adaptation(
            current,
            phx.meshing.MarkedMeshAdaptation(
                np.asarray(current.mesh.blocks[0].global_ids), hierarchy=hierarchy
            ),
            policy=policy,
        )
        adapted = phx.meshing.execute_mesh_adaptation(prepared)
        current, hierarchy = adapted.target, adapted.hierarchy
    return original, current, prepared


def test_native_cad_affine_chain_retains_chart_root_and_binds_live_target(
    native_cad_subdivision: tuple[
        phx.meshing.CellMeshingResult, phx.meshing.CellMeshingResult, phx.meshing.PreparedMeshAdaptation
    ],
) -> None:
    original, target, _ = native_cad_subdivision
    charts = target.surface_source
    assert charts is not None and original.surface_source is not None
    assert target.certification is not None and target.certification.passed
    assert charts.source_chart_id == original.surface_source.source_chart_id
    charts.require_root(original.mesh, original.geometry)
    source = target.certification.request.source
    assert isinstance(source, SurfaceSubdivisionBoundarySource)
    assert len(source.lineages) == 2
    assert isinstance(source.support.original, phx.meshing.CellMeshingResult)
    assert source.support.original.result_id == original.result_id
    assert source.support.root_topology_id == original.mesh.topology_id
    assert source.support.root_geometry_id == original.audit.geometry_id
    charts.require_bound(target.mesh, target.geometry, source)
    assert target.mesh.entity_set(2).count > original.mesh.entity_set(2).count
    assert all(association.complete for association in target.associations)
    occurrences = source.current_boundary_source
    assert occurrences.boundary_cell_ids is not None
    assert occurrences.boundary_occurrences is not None
    np.testing.assert_array_equal(
        occurrences.boundary_cell_ids,
        np.sort(np.concatenate([np.asarray(block.global_ids) for block in target.mesh.blocks])),
    )
    assert occurrences.boundary_occurrences.shape == (target.mesh.entity_set(2).count, 3, 3)
    occurrences.require_current()


def test_native_cad_affine_chain_refuses_initial_source_bound_to_target(
    native_cad_subdivision: tuple[
        phx.meshing.CellMeshingResult, phx.meshing.CellMeshingResult, phx.meshing.PreparedMeshAdaptation
    ],
) -> None:
    _, target, _ = native_cad_subdivision
    charts = target.surface_source
    assert charts is not None
    with pytest.raises(ValueError, match="SurfaceSourceCharts.root"):
        charts.require_bound(target.mesh, target.geometry, charts.boundary_source)


@pytest.mark.parametrize("shortfall", (0, 1), ids=("R", "R-minus-one"))
def test_native_cad_subdivision_measured_work_boundary(
    native_cad_subdivision: tuple[
        phx.meshing.CellMeshingResult, phx.meshing.CellMeshingResult, phx.meshing.PreparedMeshAdaptation
    ],
    shortfall: int,
) -> None:
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus

    _, _, prepared = native_cad_subdivision
    # Measured complete bounded request, including source-metadata publication.
    # The 2M discovery request uses 866043; its distinct limit/receipt binding
    # is not the bounded request's work identity.
    required = 866_045
    assert 0 < required < prepared.policy.limits.maximum_work_units
    policy = phx.meshing.MeshAdaptationPolicy(
        prepared.policy.route,
        limits=phx.meshing.MeshingLimits(
            maximum_work_units=required - shortfall,
            maximum_geometry_queries=prepared.policy.limits.maximum_geometry_queries,
            maximum_scratch_bytes=prepared.policy.limits.maximum_scratch_bytes,
        ),
        compatibility=prepared.policy.compatibility,
        association_transfer=prepared.policy.association_transfer,
    )
    print(f"CAD subdivision measured work R={required}, shortfall={shortfall}")
    if shortfall:
        with pytest.raises(MeshcoreError) as refused:
            phx.meshing.execute_mesh_adaptation(
                phx.meshing.prepare_mesh_adaptation(
                    prepared.source, prepared.request, policy=policy
                )
            )
        assert refused.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    else:
        result = phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                prepared.source, prepared.request, policy=policy
            )
        )
        assert result.target.certification is not None and result.target.certification.passed
        assert result.target.execution_evidence is not None
        assert int(np.asarray(result.target.execution_evidence.total_work_units)) == required



def test_brep_association_refinement_preserves_vertex_edge_face_classes() -> None:
    projection = plate_projection()
    mesh = plate_mesh((1.2, 1.2))
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    source = phx.meshing.certify_cell_mesh(
        mesh, CURVING_COORDINATE_CONTRACT, associations=(association,)
    )
    adapted = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(np.asarray(mesh.blocks[0].global_ids)),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
                association_transfer=phx.meshing.BRepAssociationTransfer(
                    projection, policy=CURVING_ASSOCIATION_POLICY
                ),
            ),
        )
    )
    target = adapted.target.associations[0]
    target_mesh = adapted.target.mesh
    source_classes, _ = association_classes(association, mesh)
    classes, residuals = association_classes(target, target_mesh)
    hole = source_classes[coordinate_rows(mesh, [[0.0, 1.0]])[0]]
    bottom = source_classes[coordinate_rows(mesh, [[0.0, -2.0]])[0]]
    corner = source_classes[coordinate_rows(mesh, [[2.0, -2.0]])[0]]

    assert target.complete
    assert target.provenance is phx.meshing.GeometryAssociationProvenance.LINEAGE
    assert target.parent_association_id == association.association_id
    # Corner vertices keep their B-Rep vertex.
    np.testing.assert_array_equal(
        classes[coordinate_rows(target_mesh, [[2.0, -2.0]])[0]], corner
    )
    # Children of a hole chord inherit the hole edge; the chord sagitta is the residual.
    chord = coordinate_rows(target_mesh, [[0.5, 0.5], [-0.5, -0.5]])
    np.testing.assert_array_equal(classes[chord], np.stack((hole, hole)))
    np.testing.assert_allclose(residuals[chord], 1.0 - 2**-0.5)
    # Children of straight boundary edges remain on the same edge.
    side = coordinate_rows(target_mesh, [[1.0, -2.0]])
    np.testing.assert_array_equal(classes[side], bottom[None])
    np.testing.assert_allclose(
        residuals[side], 0.0, atol=CURVING_COORDINATE_ROUNDOFF_ATOL
    )
    # Interior children inherit the face.
    points = np.asarray(target_mesh.coordinates)
    interior = (np.abs(points).sum(axis=1) > 1.0 + BOUNDARY_CLASSIFICATION_MARGIN) & (
        np.max(np.abs(points), axis=1) < 2.0 - BOUNDARY_CLASSIFICATION_MARGIN
    )
    assert np.all(classes[interior, 0] == 2)
    assert np.all(classes[~interior, 0] < 2)


def test_brep_association_collapse_rejects_cross_class_identification() -> None:
    projection = plate_projection()
    mesh = plate_mesh((1.2, 1.2))
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    removed, corner, hole = coordinate_rows(mesh, [[0.0, -2.0], [2.0, -2.0], [1.0, 0.0]])

    target, lineage = collapsed_mesh(mesh, removed, corner)
    legal = phx.meshing.propagate_association(
        association, lineage, mesh, target, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    assert legal.complete
    target, lineage = collapsed_mesh(mesh, removed, hole)
    with pytest.raises(phx.meshing.AssociationPropagationError) as error:
        phx.meshing.propagate_association(
            association,
            lineage,
            mesh,
            target,
            projection,
            policy=CURVING_ASSOCIATION_POLICY,
        )
    assert error.value.target_ids.tolist() == [int(mesh.vertex_global_ids[hole])]


def test_brep_association_collapse_keeps_true_corner_protected() -> None:
    projection = plate_projection()
    mesh = plate_mesh((1.2, 1.2))
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    corner, edge = coordinate_rows(mesh, [[2.0, -2.0], [0.0, -2.0]])
    assert np.asarray(association.source_dimensions)[corner] == 0
    target, lineage = collapsed_mesh(mesh, corner, edge)
    with pytest.raises(phx.meshing.AssociationPropagationError) as error:
        phx.meshing.propagate_association(
            association,
            lineage,
            mesh,
            target,
            projection,
            policy=CURVING_ASSOCIATION_POLICY,
        )
    assert error.value.target_ids.tolist() == [int(mesh.vertex_global_ids[edge])]


def test_brep_association_rederivation_preserves_shared_vertex_classes() -> None:
    projection = plate_projection()
    source_mesh = plate_mesh((1.2, 1.2))
    association = phx.meshing.associate_mesh_vertices(
        source_mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    source = phx.meshing.certify_cell_mesh(
        source_mesh, CURVING_COORDINATE_CONTRACT, associations=(association,)
    )
    target = phx.meshing.certify_cell_mesh(
        plate_mesh((1.3, -1.1)), CURVING_COORDINATE_CONTRACT
    )

    derived = phx.meshing.rederive_association(
        association, source, target, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    shared = np.asarray(source_mesh.coordinates)[:12]
    classes, _ = association_classes(derived, target.mesh)
    source_classes, _ = association_classes(association, source_mesh)

    assert derived.complete
    assert derived.provenance is phx.meshing.GeometryAssociationProvenance.CLASSIFICATION
    np.testing.assert_array_equal(
        classes[coordinate_rows(target.mesh, shared)],
        source_classes[coordinate_rows(source_mesh, shared)],
    )


def test_native_source_order_realization_certifies_continuous_geometry_improvement() -> (
    None
):
    model = phx.geometry.brep_sphere(
        1.0,
        coordinate_contract=CURVING_COORDINATE_CONTRACT,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    projection = phx.geometry.prepare_brep_projection(model)
    mesh = icosphere(1)
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    policy = phx.meshing.HighOrderCurvingPolicy(
        degree=10,
        relaxation_rounds=0,
        residual_tolerance=1e-7,
        fidelity_tolerance=1e-7,
    )
    result = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)

    assert result.status is CURVING_STATUS.CURVED
    assert result.evidence.accepted
    assert result.evidence.fidelity is not None
    assert result.evidence.fidelity.status == "certified"
    assert result.evidence.fidelity.mesh_to_source_upper <= policy.fidelity_tolerance
    assert result.evidence.fidelity.source_to_mesh_upper <= policy.fidelity_tolerance
    assert result.evidence.source_revision == model.source_revision
    parameters = jnp.asarray(((0.2, 0.2), (0.1, 0.7), (1 / 3, 1 / 3)), dtype=jnp.float64)

    def radial_error(geometry: CellGeometrySpec) -> float:
        elements, routes, coordinates = geometry.resolve(mesh)
        errors = []
        for element, route in zip(elements, routes, strict=True):
            if not isinstance(element, FiniteElementSpec):
                raise TypeError(
                    "curving_reference.element must be a scalar finite element; got another coordinate element"
                )
            basis, _ = element.tabulate(parameters)
            points = np.asarray(basis)[None] @ np.asarray(coordinates)[np.asarray(route)]
            errors.append(float(np.max(np.abs(np.linalg.norm(points, axis=-1) - 1.0))))
        return max(errors)

    assert radial_error(result.geometry) <= policy.fidelity_tolerance
    assert radial_error(result.geometry) < 1e-3 * radial_error(result.straight)


def test_curving_requires_explicit_continuous_approximation_budget() -> None:
    with pytest.raises(TypeError, match="fidelity_tolerance"):
        phx.meshing.HighOrderCurvingPolicy(degree=2)  # ty: ignore[missing-argument]


def test_native_projection_cannot_accept_nodal_evidence_as_continuous_fidelity() -> None:
    projection = sphere_projection()
    mesh = icosphere(1)
    policy = phx.meshing.HighOrderCurvingPolicy(
        fidelity_tolerance=1e-7, degree=2, relaxation_rounds=0
    )
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    result = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)
    assert isinstance(projection, phx.geometry.NativeBRepProjection)
    assert result.status is CURVING_STATUS.ROLLED_BACK_CERTIFICATION
    assert result.candidate is not None
    assert result.candidate.certificate.all_certified
    assert result.candidate.maximum_residual <= policy.residual_tolerance
    assert result.candidate_geometry is not None
    result.candidate.certificate.require_bound(result.candidate_geometry, mesh=mesh)
    assert sphere_error(result.candidate_geometry, mesh) > policy.fidelity_tolerance
    assert sphere_error(result.candidate_geometry, mesh) < 0.05 * sphere_error(
        result.straight, mesh
    )
    assert "source_fidelity" in result.candidate.certification_failures
    assert not result.candidate.accepted
    assert result.accepted_round is None
    np.testing.assert_array_equal(
        result.geometry.coordinates, result.straight.coordinates
    )


def test_converged_local_repair_retains_rollback_without_source_certificate() -> None:
    projection = plate_projection()
    mesh = plate_mesh((0.8, 0.8))
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )

    def curve(rounds: int, steps: int) -> HighOrderCurvingResult:
        policy = phx.meshing.HighOrderCurvingPolicy(
            fidelity_tolerance=1e-7,
            degree=2,
            relaxation_rounds=rounds,
            termination=phx.optim.OptimizationTermination(maximum_steps=steps),
        )
        return phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)

    projected = curve(0, 64)
    capped = curve(2, 64)
    relaxed = curve(2, 512)

    # Snapping the hole chord onto the arc folds the element at a hole vertex.
    assert projected.status is CURVING_STATUS.ROLLED_BACK_INVALID
    assert projected.candidate is not None
    assert projected.candidate.certificate.invalid_count > 0
    np.testing.assert_array_equal(
        projected.geometry.coordinates, projected.straight.coordinates
    )
    # Local success does not replace missing continuous evidence.
    assert capped.status is CURVING_STATUS.ROLLED_BACK_CERTIFICATION
    assert capped.candidate is not None
    assert capped.accepted_round is None and not capped.candidate.accepted
    assert phx.optim.OptimizationStatus.SUCCESS not in capped.relaxation_statuses
    np.testing.assert_array_equal(
        capped.geometry.coordinates, capped.straight.coordinates
    )
    assert relaxed.status is CURVING_STATUS.ROLLED_BACK_CERTIFICATION
    assert phx.optim.OptimizationStatus.SUCCESS in relaxed.relaxation_statuses
    assert relaxed.candidate is not None
    assert relaxed.candidate.certificate.all_certified
    assert relaxed.candidate.minimum_scaled_jacobian > 0.0
    assert "source_fidelity" in relaxed.candidate.certification_failures
    np.testing.assert_array_equal(
        relaxed.geometry.coordinates, relaxed.straight.coordinates
    )


@pytest.mark.parametrize(
    ("rounds", "accept"),
    ((0, False), (1, False), (1, True)),
    ids=("projection", "nonconverged-refusal", "nonconverged-opt-in"),
)
def test_nonconvergence_opt_in_does_not_bypass_continuous_certificate(
    rounds: int,
    accept: bool,
) -> None:
    projection = sphere_projection()
    mesh = icosphere(1)
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    policy = phx.meshing.HighOrderCurvingPolicy(
        fidelity_tolerance=1e-7,
        degree=2,
        relaxation_rounds=rounds,
        termination=phx.optim.OptimizationTermination(maximum_steps=1),
        accept_valid_nonconverged_relaxation=accept,
    )
    result = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)
    assert result.status is CURVING_STATUS.ROLLED_BACK_CERTIFICATION
    assert result.accepted_round is None
    assert result.candidate is not None
    assert result.candidate.certificate.all_certified
    assert not result.candidate.accepted
    np.testing.assert_array_equal(
        result.geometry.coordinates, result.straight.coordinates
    )
    expected = (
        () if rounds == 0 else (phx.optim.OptimizationStatus.MAXIMUM_STEPS_REACHED,)
    )
    assert result.relaxation_statuses == expected


@pytest.mark.parametrize("all_invalid", (False, True), ids=("earlier-accepted", "rollback"))
def test_newest_first_proofs_keep_the_invalid_last_candidate_and_real_fallback(
    monkeypatch: pytest.MonkeyPatch, all_invalid: bool,
) -> None:
    from phydrax.meshing import _curving as curving

    projection = sphere_projection()
    mesh = icosphere(0)
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    straight = straight_geometry(mesh, 2)
    _, routes, _ = straight.resolve(mesh)
    first_cell = np.asarray(routes[0])[0]
    project = curving._project_constrained
    projections = 0

    def projected(*args: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal projections
        coordinates, rows, projectors = project(*args)
        projections += 1
        if all_invalid or projections == 3:
            classes = args[2]
            free = [int(row) for row in first_cell if not classes.fixed[row]]
            assert len(free) >= 2
            coordinates = coordinates.copy()
            coordinates[free[0]] = coordinates[free[1]]
        return coordinates, rows, projectors

    monkeypatch.setattr(curving, "_project_constrained", projected)
    evaluate = curving._evidence
    evaluated = []

    def evidence(
        mesh: phx.discretization.CellMesh,
        geometry: CellGeometrySpec,
        *args: Any,
        **kwargs: Any,
    ) -> curving.CurvedGeometryEvidence:
        value = evaluate(mesh, geometry, *args, **kwargs)
        evaluated.append((geometry, value))
        return value

    monkeypatch.setattr(curving, "_evidence", evidence)
    result = phx.meshing.curve_cell_mesh(
        mesh, association, projection,
        policy=phx.meshing.HighOrderCurvingPolicy(
            degree=2, relaxation_rounds=2, fidelity_tolerance=0.5,
            termination=phx.optim.OptimizationTermination(maximum_steps=1),
            accept_valid_nonconverged_relaxation=True,
        ),
    )
    assert projections == 3 and len(result.minimizations) == 2
    assert result.candidate is evaluated[0][1]
    assert result.candidate_geometry is evaluated[0][0]
    assert not result.candidate.accepted
    assert not result.candidate.certificate.all_certified
    if all_invalid:
        assert result.status is CURVING_STATUS.ROLLED_BACK_INVALID
        assert result.accepted_round is None
        assert not any(value.accepted for _, value in evaluated[:-1])
        np.testing.assert_array_equal(result.geometry.coordinates, result.straight.coordinates)
    else:
        assert result.status is CURVING_STATUS.CURVED
        assert result.accepted_round == 1
        assert result.evidence is evaluated[1][1]
        assert result.evidence.accepted
        assert len(evaluated) == 2


def test_existing_coordinate_node_budget_uses_actual_supplied_layout() -> None:

    projection = plate_projection()
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((1.25, 1.25), (1.75, 1.25), (1.25, 1.75)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    geometry = straight_geometry(mesh, 2)
    evidence = phx.meshing.verify_curved_geometry(
        geometry,
        mesh,
        association,
        projection,
        policy=phx.meshing.HighOrderCurvingPolicy(
            fidelity_tolerance=1e-7, degree=2, maximum_geometry_nodes=6
        ),
    )
    assert evidence.certificate.certified_valid_count == 1
    assert evidence.node_dimensions.shape == (6,)
    with pytest.raises(ValueError, match="node resource limit"):
        phx.meshing.verify_curved_geometry(
            geometry,
            mesh,
            association,
            projection,
            policy=phx.meshing.HighOrderCurvingPolicy(
                fidelity_tolerance=1e-7, degree=2, maximum_geometry_nodes=5
            ),
        )

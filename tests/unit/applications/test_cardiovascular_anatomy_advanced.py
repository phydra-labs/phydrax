#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


cv = phx.applications.cardiovascular


def _field_space(name, count):
    vector_space = phx.linalg.ArraySpace((count,), dtype=jnp.float32)
    layout = phx.discretization.TensorDofLayout(("point",), (count,))
    return phx.discretization.DiscreteFieldSpace(
        name,
        f"{name}-support",
        layout,
        vector_space,
        representation="point_value",
        conformity="H1",
    )


def _cardiac_transfer(*, source_covered=None, matrix=None):
    source = _field_space("source-voltage", 2)
    target = _field_space("target-voltage", 3)
    transfer_matrix = (
        jnp.asarray(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0)), dtype=jnp.float32)
        if matrix is None
        else jnp.asarray(matrix, dtype=jnp.float32)
    )
    operator = phx.linalg.DenseLinearOperator(
        transfer_matrix,
        source=source.vector_space,
        target=target.vector_space,
    )
    adjoint = phx.linalg.DenseLinearOperator(
        transfer_matrix.T,
        source=target.vector_space,
        target=source.vector_space,
    )
    transfer = phx.discretization.FieldTransfer(
        source,
        target,
        operator,
        adjoint_operator=adjoint,
        properties=phx.discretization.TransferProperties(
            constant_preserving=True,
            adjoint_paired=True,
        ),
    )
    configuration = cv.anatomy.CardiacTransferConfiguration.for_transfer(
        transfer,
        "transmembrane-voltage",
        "mV",
        "source-material-reference",
        "target-material-reference",
        component_axes=(),
    )
    return cv.anatomy.CardiacFieldTransfer(
        transfer,
        configuration,
        cv.anatomy.CardiacTransferEpoch(4, 7, 2, 3),
        source_covered=source_covered,
    )


def _image_identities():
    return (
        cv.anatomy.ImageAcquisitionIdentity(
            "acq-deid-17", "series-deid-4", "MR", "cine-short-axis"
        ),
        cv.anatomy.ImageDeidentificationIdentity(
            "dicom-basic-profile", "deid-run-22", "attestation-22"
        ),
        cv.anatomy.ImageDataRightsIdentity(
            "rights-17",
            "license-clinical-research",
            "controller-site-a",
            permitted_use_ids=("geometry-reconstruction", "model-validation"),
        ),
    )


def _image_metadata(affine, *, coordinate_frame, host_fields=None):
    acquisition, deidentification, rights = _image_identities()
    return cv.anatomy.CardiacImageBoundaryMetadata(
        affine,
        acquisition,
        deidentification,
        rights,
        coordinate_frame=coordinate_frame,
        host_fields={} if host_fields is None else host_fields,
    )


def _pmj_geometry():
    graph = jnp.asarray(
        (
            (0.0, 0.25, 0.0),
            (9.0, 0.0, 0.0),
            (4.0, 5.0, 0.0),
            (7.0, 7.0, 0.0),
        ),
        dtype=jnp.float32,
    )
    myocardial = jnp.asarray(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (4.0, 4.0, 0.0),
        ),
        dtype=jnp.float32,
    )
    return graph, myocardial


def _prepared_pmj(*, capacity=4, maximum_distance=2.0, candidate_mask=None):
    graph, myocardial = _pmj_geometry()
    mask = (
        jnp.asarray((True, True, False, False))
        if candidate_mask is None
        else candidate_mask
    )
    epoch = cv.anatomy.PMJAttachmentEpoch(5, 8)
    prepared = cv.anatomy.PurkinjeAttachmentPlan(capacity, maximum_distance).prepare(
        graph,
        myocardial,
        pmj_candidate_mask=mask,
        graph_geometry_id="purkinje-graph-v1",
        myocardial_geometry_id="myocardial-support-v3",
        epoch=epoch,
    )
    return prepared, graph, myocardial, epoch


def test_cardiac_transfer_reports_coverage_constant_adjoint_and_configuration():
    transfer = _cardiac_transfer()
    epoch = cv.anatomy.CardiacTransferEpoch(4, 7, 2, 3)
    result = transfer.apply(
        jnp.asarray((2.0, 4.0), dtype=jnp.float32),
        epoch,
        configuration_id=transfer.configuration.configuration_id,
    )

    assert jnp.allclose(result.value, jnp.asarray((2.0, 3.0, 4.0)))
    assert bool(result.evidence.coverage_complete)
    assert result.evidence.source_coverage_fraction == 1.0
    assert result.evidence.target_coverage_fraction == 1.0
    assert result.evidence.constant_error <= 1.0e-7
    assert result.evidence.adjoint_error <= transfer.adjoint_tolerance
    assert bool(result.evidence.constant_preserved)
    assert bool(result.evidence.adjoint_consistent)
    assert bool(result.evidence.configuration_matches)
    assert bool(result.evidence.accepted)


@pytest.mark.parametrize(
    "epoch",
    (
        (5, 7, 2, 3),
        (4, 8, 2, 3),
        (4, 7, 3, 3),
        (4, 7, 2, 4),
    ),
)
def test_cardiac_transfer_invalidates_every_geometry_and_reference_epoch(epoch):
    transfer = _cardiac_transfer()
    result = transfer.apply(
        jnp.asarray((2.0, 4.0), dtype=jnp.float32),
        cv.anatomy.CardiacTransferEpoch(*epoch),
        configuration_id=transfer.configuration.configuration_id,
    )

    assert not bool(result.evidence.epoch_matches)
    assert not bool(result.evidence.accepted)
    assert jnp.allclose(result.value, jnp.asarray((2.0, 3.0, 4.0)))


def test_cardiac_transfer_fails_closed_for_configuration_coverage_and_claims():
    epoch = cv.anatomy.CardiacTransferEpoch(4, 7, 2, 3)
    partial = _cardiac_transfer(source_covered=jnp.asarray((True, False)))
    partial_result = partial.apply(
        jnp.asarray((1.0, 2.0), dtype=jnp.float32),
        epoch,
        configuration_id=partial.configuration.configuration_id,
    )
    wrong_configuration = _cardiac_transfer()
    configuration_result = wrong_configuration.apply(
        jnp.asarray((1.0, 2.0), dtype=jnp.float32),
        epoch,
        configuration_id="another-configuration",
    )
    nonconstant = _cardiac_transfer(matrix=((1.0, 0.0), (0.5, 0.25), (0.0, 1.0)))
    constant_result = nonconstant.apply(
        jnp.asarray((1.0, 2.0), dtype=jnp.float32),
        epoch,
        configuration_id=nonconstant.configuration.configuration_id,
    )

    assert not bool(partial_result.evidence.coverage_complete)
    assert partial_result.evidence.source_coverage_fraction == 0.5
    assert not bool(partial_result.evidence.accepted)
    assert not bool(configuration_result.evidence.configuration_matches)
    assert not bool(configuration_result.evidence.accepted)
    assert not bool(constant_result.evidence.constant_preserved)
    assert not bool(constant_result.evidence.accepted)


def test_image_boundary_preserves_affine_and_explicitly_converts_lps_ras():
    matrix = jnp.asarray(
        (
            (1.25, 0.0, 0.0, 15.0),
            (0.0, 1.5, 0.0, -20.0),
            (0.0, 0.0, 2.0, 6.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    affine = cv.anatomy.MedicalImageAffine(
        matrix,
        cv.anatomy.ImageCoordinateFrame.LPS,
        cv.anatomy.ImageLengthUnit.MILLIMETER,
    )
    metadata = _image_metadata(
        affine,
        coordinate_frame=cv.anatomy.ImageCoordinateFrame.LPS,
        host_fields={"field_strength_t": 3.0, "sequence_id": "cine-bSSFP"},
    )
    ras = metadata.reframe(cv.anatomy.ImageCoordinateFrame.RAS)
    restored = ras.reframe(cv.anatomy.ImageCoordinateFrame.LPS)

    assert jnp.array_equal(metadata.affine.voxel_to_world, matrix)
    assert jnp.array_equal(
        ras.affine.voxel_to_world[:2], -metadata.affine.voxel_to_world[:2]
    )
    assert jnp.array_equal(
        ras.affine.voxel_to_world[2:], metadata.affine.voxel_to_world[2:]
    )
    assert jnp.array_equal(restored.affine.voxel_to_world, matrix)
    assert metadata.acquisition.acquisition_id == "acq-deid-17"
    assert metadata.deidentification.attestation_id == "attestation-22"
    assert metadata.data_rights.rights_id == "rights-17"


def test_image_boundary_converts_units_with_exact_kernel_scale():
    affine = cv.anatomy.MedicalImageAffine(
        jnp.asarray(
            (
                (0.1, 0.0, 0.0, 1.0),
                (0.0, 0.2, 0.0, 2.0),
                (0.0, 0.0, 0.3, 3.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
        cv.anatomy.ImageCoordinateFrame.RAS,
        cv.anatomy.ImageLengthUnit.CENTIMETER,
    )
    kernel = _image_metadata(
        affine, coordinate_frame=cv.anatomy.ImageCoordinateFrame.RAS
    ).in_kernel_units()

    assert kernel.affine.length_unit is cv.anatomy.ImageLengthUnit.MILLIMETER
    assert jnp.array_equal(
        kernel.affine.voxel_to_world,
        jnp.asarray(
            (
                (1.0, 0.0, 0.0, 10.0),
                (0.0, 2.0, 0.0, 20.0),
                (0.0, 0.0, 3.0, 30.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
    )


def test_image_boundary_rejects_conflicting_frames_and_phi_fields():
    affine = cv.anatomy.MedicalImageAffine(
        jnp.eye(4),
        cv.anatomy.ImageCoordinateFrame.LPS,
        cv.anatomy.ImageLengthUnit.MILLIMETER,
    )
    with pytest.raises(ValueError, match="conflicts"):
        _image_metadata(affine, coordinate_frame=cv.anatomy.ImageCoordinateFrame.RAS)
    with pytest.raises(ValueError, match="non-PHI allowlist"):
        _image_metadata(
            affine,
            coordinate_frame=cv.anatomy.ImageCoordinateFrame.LPS,
            host_fields={"PatientName": "not-admitted"},
        )
    with pytest.raises(ValueError, match="non-PHI allowlist"):
        _image_metadata(
            affine,
            coordinate_frame=cv.anatomy.ImageCoordinateFrame.LPS,
            host_fields={"acquisition_date": "not-admitted"},
        )


def test_pmj_preparation_is_deterministic_fixed_shape_and_lowest_index_tied():
    first, graph, myocardial, epoch = _prepared_pmj()
    second, _, _, _ = _prepared_pmj()
    candidate = first.evaluate(graph, myocardial, epoch)

    assert first.graph_indices.shape == (4,)
    assert first.myocardial_support_indices.shape == (4,)
    assert first.route_active.shape == (4,)
    assert jnp.array_equal(first.graph_indices, jnp.asarray((0, 1, 0, 0)))
    assert jnp.array_equal(first.myocardial_support_indices, jnp.asarray((0, 2, 0, 0)))
    assert jnp.array_equal(first.graph_indices, second.graph_indices)
    assert jnp.array_equal(
        first.myocardial_support_indices, second.myocardial_support_indices
    )
    assert first.attachment_id == second.attachment_id
    assert candidate.graph_points_mm.shape == (4, 3)
    assert candidate.myocardial_points_mm.shape == (4, 3)
    assert candidate.evidence.attached_count == 2
    assert candidate.evidence.capacity_remaining == 2
    assert candidate.evidence.coverage_fraction == 1.0
    assert candidate.evidence.fixed_routes
    assert bool(candidate.evidence.accepted)


def test_pmj_preparation_refuses_capacity_overflow_and_missing_support():
    graph, myocardial = _pmj_geometry()
    epoch = cv.anatomy.PMJAttachmentEpoch(5, 8)
    plan = cv.anatomy.PurkinjeAttachmentPlan(2, 2.0)
    with pytest.raises(ValueError, match="exceeds configured capacity"):
        plan.prepare(
            graph,
            myocardial,
            pmj_candidate_mask=jnp.asarray((True, True, True, False)),
            graph_geometry_id="graph",
            myocardial_geometry_id="myocardium",
            epoch=epoch,
        )
    with pytest.raises(ValueError, match="active myocardial support"):
        plan.prepare(
            graph,
            myocardial,
            pmj_candidate_mask=jnp.asarray((True, False, False, False)),
            myocardial_active_mask=jnp.zeros((4,), dtype=bool),
            graph_geometry_id="graph",
            myocardial_geometry_id="myocardium",
            epoch=epoch,
        )


def test_pmj_runtime_never_falls_back_to_a_new_nearest_support():
    prepared, graph, myocardial, epoch = _prepared_pmj()
    moved = myocardial.at[0].set(jnp.asarray((20.0, 0.0, 0.0), dtype=myocardial.dtype))
    moved = moved.at[1].set(jnp.asarray((0.0, 0.25, 0.0), dtype=myocardial.dtype))
    candidate = prepared.evaluate(graph, moved, epoch)

    assert prepared.myocardial_support_indices[0] == 0
    assert candidate.evidence.distances_mm[0] > 2.0
    assert not bool(candidate.evidence.within_distance[0])
    assert candidate.evidence.uncovered_count == 1
    assert not bool(candidate.evidence.accepted)


def test_pmj_attachment_invalidates_epoch_and_differentiates_fixed_routes_only():
    prepared, graph, myocardial, epoch = _prepared_pmj()
    stale = prepared.evaluate(
        graph,
        myocardial,
        cv.anatomy.PMJAttachmentEpoch(6, 8),
    )

    def fixed_route_distance(points):
        candidate = prepared.evaluate(points, myocardial, epoch)
        return jnp.sum(candidate.evidence.distances_mm)

    gradient = jax.grad(fixed_route_distance)(graph)

    assert not bool(stale.evidence.epoch_matches)
    assert not bool(stale.evidence.accepted)
    assert gradient.shape == graph.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(gradient[2:] == 0.0)
    assert jnp.array_equal(prepared.myocardial_support_indices, jnp.asarray((0, 2, 0, 0)))


def test_pmj_gather_and_scatter_preserve_fixed_capacity_and_accumulate_routes():
    prepared, _, _, _ = _prepared_pmj()
    graph_values = jnp.asarray((2.0, 3.0, 5.0, 7.0))
    myocardial_values = jnp.asarray((11.0, 13.0, 17.0, 19.0))

    gathered_graph = prepared.gather_graph(graph_values)
    gathered_myocardium = prepared.gather_myocardium(myocardial_values)
    scattered = prepared.scatter_to_myocardium(jnp.asarray((1.5, 2.5, 100.0, 100.0)))

    assert jnp.array_equal(gathered_graph, jnp.asarray((2.0, 3.0, 0.0, 0.0)))
    assert jnp.array_equal(gathered_myocardium, jnp.asarray((11.0, 17.0, 0.0, 0.0)))
    assert jnp.array_equal(scattered, jnp.asarray((1.5, 0.0, 2.5, 0.0)))


def _high_order_geometry(cell_kind, *, degree=2):
    if cell_kind == "tetrahedron":
        mesh_coordinates = jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        vertices = jnp.asarray(((0, 1, 2, 3),))
    else:
        mesh_coordinates = jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            )
        )
        vertices = jnp.asarray(((0, 1, 2, 3, 4, 5, 6, 7),))
    block = phx.discretization.CellBlock("myocardium", cell_kind, vertices)
    mesh = phx.discretization.CellMesh(mesh_coordinates, (block,))
    element = phx.discretization.fem.lagrange_element(cell_kind, degree)
    coordinate_spec = phx.discretization.fem.FiniteElementCoordinateSpec(
        {"myocardium": element},
        {"myocardium": jnp.arange(element.local_dof_count)[None, :]},
        element.reference_nodes,
    )
    profile = cv.anatomy.CardiacBoundaryProfile(
        "ventricular-volume",
        required_roles=("epicardium",),
    )
    epoch = cv.anatomy.HighOrderGeometryEpoch(3, 2)
    plan = cv.anatomy.HighOrderCardiacGeometryPlan(
        mesh,
        coordinate_spec,
        boundary_role_id="ventricular-boundary-roles-v1",
        boundary_profile=profile,
        prepared_epoch=epoch,
    )
    return plan, coordinate_spec, epoch


@pytest.mark.parametrize(
    ("cell_kind", "expected_measure", "quadrature_order"),
    (("tetrahedron", 1.0 / 6.0, 5), ("hexahedron", 1.0, 4)),
)
def test_high_order_p2_q2_geometry_qualifies_jacobian_and_measure(
    cell_kind, expected_measure, quadrature_order
):
    plan, coordinate_spec, epoch = _high_order_geometry(cell_kind)
    prepared = plan.prepare()
    reference_coordinates = coordinate_spec.coordinates
    curved_coordinates = reference_coordinates.at[:, 2].add(
        0.05 * reference_coordinates[:, 0] * (1.0 - reference_coordinates[:, 0])
    )
    candidate = prepared.evaluate(
        curved_coordinates,
        epoch,
        boundary_role_id=plan.boundary_role_id,
        boundary_profile_id=plan.boundary_profile.profile_id,
    )

    def integrated_measure(coordinates):
        result = prepared.evaluate(
            coordinates,
            epoch,
            boundary_role_id=plan.boundary_role_id,
            boundary_profile_id=plan.boundary_profile.profile_id,
        )
        return jnp.sum(result.block_cell_measures_mm3[0])

    gradient = jax.grad(integrated_measure)(curved_coordinates)

    assert prepared.quadrature_orders == (quadrature_order,)
    assert bool(candidate.evidence.fixed_topology)
    assert bool(candidate.evidence.orientation_valid)
    assert bool(candidate.evidence.measure_valid)
    assert bool(candidate.evidence.accepted)
    assert not jnp.array_equal(curved_coordinates, reference_coordinates)
    assert jnp.allclose(
        candidate.block_cell_measures_mm3[0],
        jnp.asarray((expected_measure,)),
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert gradient.shape == coordinate_spec.coordinates.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_high_order_geometry_rejects_inversion_without_runtime_fallback():
    plan, coordinate_spec, epoch = _high_order_geometry("tetrahedron")
    prepared = plan.prepare()
    inverted = coordinate_spec.coordinates.at[:, 0].multiply(-1.0)
    candidate = prepared.evaluate(
        inverted,
        epoch,
        boundary_role_id=plan.boundary_role_id,
        boundary_profile_id=plan.boundary_profile.profile_id,
    )

    assert jnp.array_equal(candidate.coordinates_mm, inverted)
    assert not bool(candidate.evidence.orientation_valid)
    assert not bool(candidate.evidence.accepted)
    assert not bool(candidate.evidence.transfer_required)
    assert not bool(candidate.evidence.rebuild_required)


def test_high_order_geometry_distinguishes_transfer_and_rebuild_evidence():
    plan, coordinate_spec, _ = _high_order_geometry("hexahedron")
    prepared = plan.prepare()
    reference_change = prepared.evaluate(
        coordinate_spec.coordinates,
        cv.anatomy.HighOrderGeometryEpoch(3, 3),
        boundary_role_id=plan.boundary_role_id,
        boundary_profile_id=plan.boundary_profile.profile_id,
    )
    geometry_change = prepared.evaluate(
        coordinate_spec.coordinates,
        cv.anatomy.HighOrderGeometryEpoch(4, 2),
        boundary_role_id=plan.boundary_role_id,
        boundary_profile_id=plan.boundary_profile.profile_id,
    )
    role_change = prepared.evaluate(
        coordinate_spec.coordinates,
        plan.prepared_epoch,
        boundary_role_id="different-boundary-roles",
        boundary_profile_id=plan.boundary_profile.profile_id,
    )

    assert bool(reference_change.evidence.transfer_required)
    assert not bool(reference_change.evidence.rebuild_required)
    assert bool(geometry_change.evidence.transfer_required)
    assert bool(geometry_change.evidence.rebuild_required)
    assert bool(role_change.evidence.transfer_required)
    assert bool(role_change.evidence.rebuild_required)
    assert not bool(reference_change.evidence.accepted)
    assert not bool(geometry_change.evidence.accepted)
    assert not bool(role_change.evidence.accepted)


def test_high_order_geometry_refuses_linear_and_unsupported_volume_routes():
    with pytest.raises(ValueError, match="qualified P2 tetrahedral"):
        _high_order_geometry("tetrahedron", degree=1)

    prism_coordinates = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        )
    )
    prism_block = phx.discretization.CellBlock(
        "myocardium", "prism", jnp.asarray(((0, 1, 2, 3, 4, 5),))
    )
    prism_mesh = phx.discretization.CellMesh(prism_coordinates, (prism_block,))
    prism_element = phx.discretization.fem.lagrange_element("prism", 2)
    prism_spec = phx.discretization.fem.FiniteElementCoordinateSpec(
        {"myocardium": prism_element},
        {"myocardium": jnp.arange(prism_element.local_dof_count)[None, :]},
        prism_element.reference_nodes,
    )
    profile = cv.anatomy.CardiacBoundaryProfile(
        "ventricular-volume", required_roles=("epicardium",)
    )
    with pytest.raises(ValueError, match="P2 tetrahedra and Q2 hexahedra only"):
        cv.anatomy.HighOrderCardiacGeometryPlan(
            prism_mesh,
            prism_spec,
            boundary_role_id="prism-roles",
            boundary_profile=profile,
            prepared_epoch=cv.anatomy.HighOrderGeometryEpoch(0, 0),
        )

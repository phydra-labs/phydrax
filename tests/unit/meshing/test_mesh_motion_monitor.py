#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np

import phydrax as phx


meshing = phx.meshing
Decision = meshing.MeshMotionDecision
_POINTS = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)))
_CELLS = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32)


def _certified(points, cells):
    mesh = phx.discretization.CellMesh.from_triangles(points, cells)
    return meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _shifted_center(source, shift):
    coordinates = np.asarray(source.mesh.coordinates).copy()
    center = int(np.argmin(np.linalg.norm(coordinates - 0.5, axis=1)))
    coordinates[center, 0] += shift
    return coordinates, center


def _grid(count):
    axis = np.linspace(0.0, 1.0, count + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    index = np.arange(points.shape[0]).reshape(count + 1, count + 1)
    a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    return points, np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))


def test_monitor_escalates_deterministically_from_accept_to_relocate_to_remesh():
    source = _certified(_POINTS, _CELLS)
    monitor = meshing.MeshMotionMonitor(source.mesh)

    decisions = [
        monitor.assess(_shifted_center(source, shift)[0], boundary_residual=0.0).decision
        for shift in (0.1, 0.3, 0.4, 0.45, 0.48)
    ]

    assert decisions == [
        Decision.ACCEPT_MOTION,
        Decision.ACCEPT_MOTION,
        Decision.RELOCATE,
        Decision.REMESH,
        Decision.REMESH,
    ]
    repeated = [
        monitor.assess(_shifted_center(source, 0.4)[0], boundary_residual=0.0)
        for _ in range(2)
    ]
    assert repeated[0].assessment_id == repeated[1].assessment_id
    assert repeated[0].minimum_jacobian_ratio < 0.3
    # A folded mesh is never remeshed directly: untangling must come first.
    folded = monitor.assess(_shifted_center(source, 0.6)[0], boundary_residual=0.0)
    assert folded.decision is Decision.RELOCATE
    assert "uncertified-cells" in folded.reasons
    assert folded.minimum_jacobian_ratio < 0.0


def test_monitor_rejects_unrealized_boundaries_and_nonfinite_coordinates():
    source = _certified(_POINTS, _CELLS)
    monitor = meshing.MeshMotionMonitor(source.mesh)
    coordinates = np.asarray(source.mesh.coordinates)

    residual = monitor.assess(coordinates, boundary_residual=1.0e-3)
    nonfinite = coordinates.copy()
    nonfinite[0, 0] = np.nan

    assert residual.decision is Decision.REJECT
    assert residual.reasons == ("boundary-residual",)
    rejected = monitor.assess(nonfinite, boundary_residual=0.0)
    assert rejected.decision is Decision.REJECT
    assert rejected.certificate is None


def test_advance_untangles_a_folded_mesh_with_a_bit_identical_boundary():
    source = _certified(_POINTS, _CELLS)
    monitor = meshing.MeshMotionMonitor(source.mesh)
    folded, center = _shifted_center(source, 0.6)

    advance = meshing.advance_mesh_motion(monitor, source, folded, boundary_residual=0.0)

    assert advance.decision is Decision.RELOCATE
    assert advance.accepted
    assert [value.decision for value in advance.assessments] == [
        Decision.RELOCATE,
        Decision.ACCEPT_MOTION,
    ]
    assert advance.relocation.untangling.succeeded
    relocated = np.asarray(advance.relocation.coordinates)
    boundary = np.arange(relocated.shape[0]) != center
    np.testing.assert_array_equal(relocated[boundary], folded[boundary])
    np.testing.assert_allclose(relocated[center], (0.5, 0.5), atol=1.0e-5)
    assert advance.result.audit.passed
    assert advance.transition is None


def test_advance_requests_a_metric_remesh_when_relocation_cannot_recover():
    points, cells = _grid(4)
    source = _certified(points, cells)
    monitor = meshing.MeshMotionMonitor(source.mesh)
    squeezed = np.asarray(source.mesh.coordinates) * np.asarray((0.2, 1.0))

    advance = meshing.advance_mesh_motion(
        monitor, source, squeezed, boundary_residual=0.0
    )

    assert advance.decision is Decision.REMESH
    assert not advance.accepted
    assert advance.adaptation is None
    assert advance.assessments[0].decision in (Decision.RELOCATE, Decision.REMESH)
    assert advance.assessments[-1].decision is not Decision.ACCEPT_MOTION
    assert advance.relocation.accepted
    metric = advance.remesh_metric
    assert metric.scope.source_id == advance.result.mesh.mesh_id
    # The reference size h = 1/4 per vertex restores isotropic reference cells.
    np.testing.assert_allclose(
        np.asarray(metric.values)[:, 0, 0],
        np.asarray(metric.values)[:, 1, 1],
    )
    assert np.all(np.asarray(metric.values)[:, 0, 0] > 1.0)

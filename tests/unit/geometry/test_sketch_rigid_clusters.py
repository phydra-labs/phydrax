#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import numpy.typing as npt

from phydrax.geometry.design._reduced import _prepare_sketch_rigid_clusters
from phydrax.geometry.design._schema import ParameterId
from phydrax.geometry.design._sketch import FixedPoint, Horizontal, PointDistance, Sketch


def _triangle(points: npt.NDArray[np.float64], *, anchor: bool = False) -> Sketch:
    constraints: list[PointDistance | FixedPoint] = [
        PointDistance(0, 1, 1.0),
        PointDistance(0, 2, 1.0),
        PointDistance(1, 2, 1.0),
    ]
    if anchor:
        constraints.append(FixedPoint(0, points[0]))
    return Sketch(points, constraints=constraints)


def test_floating_triangle_pose_and_mirrored_branch() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(0.75)]], dtype=np.float64)
    sketch = _triangle(points)
    prepared = _prepare_sketch_rigid_clusters(sketch, points)
    (cluster,) = prepared.clusters
    assert cluster.point_indices == (0, 1, 2)
    assert cluster.constraint_indices == (0, 1, 2)
    assert cluster.locally_rigid_indicated
    assert cluster.root_evidence.jacobian_rank == 3
    assert cluster.fixed_point_constraint_indices == ()
    assert cluster.branch_witness == (0, 1, 2)
    assert cluster.branch_sign == 1
    exact = cluster.exact_distance_witness
    assert exact is not None
    assert exact.seed_points == (0, 1)
    assert exact.seed_constraint_index == 0
    assert len(exact.triangles) == 1
    assert exact.triangles[0].point == 2
    assert exact.triangles[0].constraint_indices == (1, 2, 0)
    assert exact.triangles[0].discriminant == 3
    assert exact.triangles[0].orientation_sign == 1
    assert prepared.retained_constraint_indices == (0, 1, 2)
    np.testing.assert_array_equal(prepared.seed_points, points)
    np.testing.assert_allclose(cluster.points_at_pose(cluster.pose_seed), points)
    moved = np.asarray(
        cluster.points_at_pose(np.array([2.0, 3.0, np.pi / 2], dtype=np.float64)),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        moved, [[2.0, 3.0], [2.0, 4.0], [2.0 - np.sqrt(0.75), 3.5]], atol=1e-12
    )
    mirrored = points.copy()
    mirrored[:, 1] *= -1
    other = _prepare_sketch_rigid_clusters(sketch, mirrored).clusters[0]
    assert other.locally_rigid_indicated
    assert other.branch_sign == -1
    other_witness = other.exact_distance_witness
    assert other_witness is not None
    assert other_witness.triangles[0].orientation_sign == -1


def test_underconstrained_and_inconsistent_remain_numerical() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    sketch = Sketch(points, constraints=(PointDistance(0, 1, 1), PointDistance(1, 2, 1)))
    prepared = _prepare_sketch_rigid_clusters(sketch, points)
    (cluster,) = prepared.clusters
    assert not cluster.locally_rigid_indicated
    assert cluster.root_evidence.jacobian_rank == 2
    assert cluster.branch_witness is None
    assert prepared.retained_constraint_indices == (0, 1)
    inconsistent = Sketch(
        points[:2], constraints=(PointDistance(0, 1, 1), PointDistance(0, 1, 2))
    )
    candidate = _prepare_sketch_rigid_clusters(inconsistent, points[:2]).clusters[0]
    assert not candidate.locally_rigid_indicated
    assert not candidate.root_evidence.root_converged


def test_anchor_unsupported_degenerate_and_dimension_edits() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    dimension = ParameterId("part", "length")
    sketch = Sketch(
        points,
        feature_id="part",
        lines=np.array([[0, 2]], dtype=np.int64),
        constraints=(
            PointDistance(0, 1, 1, parameter_id=dimension),
            FixedPoint(0, np.array([2.0, 3.0], dtype=np.float64)),
            Horizontal(0),
            PointDistance(1, 2, 0),
        ),
    )
    prepared = _prepare_sketch_rigid_clusters(sketch, points)
    assert prepared.retained_constraint_indices == (0, 1, 2, 3)
    (cluster,) = prepared.clusters
    assert cluster.point_indices == (0, 1)
    assert cluster.fixed_point_constraint_indices == (1,)
    assert cluster.locally_rigid_indicated
    np.testing.assert_allclose(prepared.seed_points, [[2.0, 3.0], [3.0, 3.0], [3.0, 0.0]])
    np.testing.assert_allclose(
        cluster.points_at_pose(cluster.pose_seed), [[2.0, 3.0], [3.0, 3.0]]
    )
    edited = _prepare_sketch_rigid_clusters(
        sketch, points, dimensions=sketch.dimensions.updated({dimension: 2.0})
    )
    assert not edited.clusters[0].root_evidence.root_converged
    assert not edited.clusters[0].locally_rigid_indicated
    np.testing.assert_array_equal(edited.seed_points, points)
    collapsed = points.copy()
    collapsed[1] = collapsed[0]
    assert _prepare_sketch_rigid_clusters(sketch, collapsed).clusters == ()


def test_multiple_anchors_and_nonrigid_component_do_not_move() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(0.75)]], dtype=np.float64)
    base = _triangle(points)
    sketch = Sketch(
        points,
        constraints=(
            *base.constraints,
            FixedPoint(0, np.array([2.0, 0.0], dtype=np.float64)),
            FixedPoint(1, np.array([5.0, 0.0], dtype=np.float64)),
        ),
    )
    prepared = _prepare_sketch_rigid_clusters(sketch, points)
    assert prepared.clusters[0].locally_rigid_indicated
    np.testing.assert_array_equal(prepared.seed_points, points)
    chain = Sketch(
        points,
        constraints=(
            PointDistance(0, 1, 1),
            PointDistance(1, 2, 1),
            FixedPoint(0, np.array([2.0, 0.0], dtype=np.float64)),
        ),
    )
    prepared = _prepare_sketch_rigid_clusters(chain, points)
    assert not prepared.clusters[0].locally_rigid_indicated
    np.testing.assert_array_equal(prepared.seed_points, points)


def test_exact_distance_triangulation_records_each_point_and_constraint() -> None:
    height = np.sqrt(0.75)
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, height], [0.5, -height]], dtype=np.float64
    )
    sketch = Sketch(
        points,
        constraints=(
            PointDistance(0, 1, 1),
            PointDistance(0, 2, 1),
            PointDistance(1, 2, 1),
            PointDistance(0, 3, 1),
            PointDistance(1, 3, 1),
        ),
    )
    cluster = _prepare_sketch_rigid_clusters(sketch, points).clusters[0]
    witness = cluster.exact_distance_witness
    assert witness is not None
    assert tuple(step.point for step in witness.triangles) == (2, 3)
    assert tuple(step.orientation_sign for step in witness.triangles) == (1, -1)
    assert witness.triangles[1].constraint_indices == (3, 4, 0)


def test_exact_distance_witness_rejects_tangent_coincident_and_collinear() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    tangent = Sketch(
        points,
        constraints=(
            PointDistance(0, 1, 2),
            PointDistance(0, 2, 1),
            PointDistance(1, 2, 1),
        ),
    )
    assert (
        _prepare_sketch_rigid_clusters(tangent, points).clusters[0].exact_distance_witness
        is None
    )
    coincident = Sketch(
        points,
        constraints=(
            PointDistance(0, 1, 0),
            PointDistance(0, 2, 1),
            PointDistance(1, 2, 1),
        ),
    )
    assert (
        _prepare_sketch_rigid_clusters(coincident, points)
        .clusters[0]
        .exact_distance_witness
        is None
    )
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    assert (
        _prepare_sketch_rigid_clusters(_triangle(collinear), collinear)
        .clusters[0]
        .exact_distance_witness
        is None
    )


def test_discriminant_uses_exact_represented_distances() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    near_two = np.nextafter(2.0, 0.0)
    sketch = Sketch(
        points,
        constraints=(
            PointDistance(0, 1, near_two),
            PointDistance(0, 2, 1),
            PointDistance(1, 2, 1),
        ),
    )
    witness = (
        _prepare_sketch_rigid_clusters(sketch, points).clusters[0].exact_distance_witness
    )
    assert witness is not None
    assert witness.triangles[0].discriminant > 0
    assert witness.triangles[0].discriminant.denominator > 1

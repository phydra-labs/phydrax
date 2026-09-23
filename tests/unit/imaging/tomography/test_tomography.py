import numpy as np
import pytest

import phydrax as phx


def _support(origin=(-1.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0)):
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-world",
        reference_frame="scanner",
    )
    rays = phx.measurement.RaySampleSupport(
        np.asarray((origin,)),
        np.asarray((direction,)),
        ("ray-0",),
        contract,
        far=np.asarray((5.0,)),
    )
    return phx.imaging.tomography.ProjectionSupport(rays, (1,), ("view-0",))


def test_voxel_projector_and_transpose_are_matched():
    support = _support()
    plan = phx.imaging.tomography.VoxelXRayTransformPlan(
        support,
        (2, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
        support.rays.coordinate_contract,
    )
    attenuation = np.asarray((1.0, 2.0)).reshape((2, 1, 1))
    projected = plan.forward(attenuation)
    np.testing.assert_allclose(projected.values, (3.0,))
    np.testing.assert_allclose(projected.evidence.path_length, (2.0,))
    np.testing.assert_allclose(
        plan.transpose(np.asarray((1.0,))).reshape((-1,)), (1.0, 1.0)
    )
    left = np.vdot(np.asarray(projected.values), np.asarray((0.7,)))
    right = np.vdot(attenuation, np.asarray(plan.transpose(np.asarray((0.7,)))))
    np.testing.assert_allclose(left, right)


def test_beer_lambert_and_iterative_reconstruction_reduce_projection_residual():
    detector = phx.imaging.tomography.BeerLambertPlan(np.asarray((100.0,)))
    response = detector.evaluate(np.asarray((np.log(2.0),)))
    np.testing.assert_allclose(response.expected_signal, (50.0,))
    fbp = phx.imaging.tomography.FilteredBackprojectionPlan(
        np.linspace(0.0, np.pi, 4, endpoint=False),
        np.linspace(-1.0, 1.0, 8),
        np.linspace(-0.5, 0.5, 4),
        np.linspace(-0.5, 0.5, 4),
    ).reconstruct(np.zeros((4, 8)))
    np.testing.assert_allclose(fbp, 0.0)
    support = _support()
    plan = phx.imaging.tomography.VoxelXRayTransformPlan(
        support,
        (2, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
        support.rays.coordinate_contract,
    )
    solved = phx.imaging.tomography.IterativeCTPlan(plan, 4).solve(np.asarray((3.0,)))
    assert bool(solved.successful)
    assert solved.residual_norms[-1] <= solved.residual_norms[0]


def test_tetrahedral_projector_has_matched_transpose():
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype="float64")
    support = _support(origin=(-1.0, 0.1, 0.1))
    transform = phx.imaging.tomography.TetrahedralXRayTransformPlan(
        support,
        vertices,
        np.asarray(((0, 1, 2, 3),)),
        support.rays.coordinate_contract,
    )
    x = np.asarray((2.0,))
    y = np.asarray((0.7,))
    np.testing.assert_allclose(
        np.vdot(transform.forward(x), y), np.vdot(x, transform.transpose(y))
    )


def test_tomography_plans_refuse_mismatched_geometry_and_invalid_policies():
    support = _support()
    wrong_contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-world",
        reference_frame="scanner",
    )
    with pytest.raises(ValueError, match="spatial coordinate contract"):
        phx.imaging.tomography.VoxelXRayTransformPlan(
            support,
            (2, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            wrong_contract,
        )
    with pytest.raises(ValueError, match="Voxel geometry"):
        phx.imaging.tomography.VoxelXRayTransformPlan(
            support,
            (2, 1, 1),
            (np.nan, 0, 0),
            (1, 1, 1),
            support.rays.coordinate_contract,
        )
    with pytest.raises(ValueError, match="at least two"):
        phx.imaging.tomography.FilteredBackprojectionPlan(
            np.asarray((0.0,)),
            np.asarray((0.0,)),
            np.asarray((0.0,)),
            np.asarray((0.0,)),
        )
    transform = phx.imaging.tomography.VoxelXRayTransformPlan(
        support,
        (2, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
        support.rays.coordinate_contract,
    )
    with pytest.raises(ValueError, match="iteration_count"):
        phx.imaging.tomography.IterativeCTPlan(transform, 0)
    with pytest.raises(ValueError, match="Detector parameters"):
        phx.imaging.tomography.BeerLambertPlan(np.asarray((1.0,)), dark_signal=np.nan)

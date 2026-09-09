import numpy as np

import phydrax as phx


def _contract():
    return phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-world",
        reference_frame="world",
    )


def test_constant_structured_index_preserves_straight_hamiltonian_rays():
    field = phx.optics.geometric.StructuredRefractiveIndexField(
        np.ones((4, 4, 4)),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        _contract(),
        field_id="uniform-index",
    )
    prepared = phx.optics.geometric.GradedIndexRayPlan(field, 0.1, 10).prepare()
    result = prepared.integrate(
        np.asarray(((0.5, 0.5, 0.5),)),
        np.asarray(((1.0, 0.0, 0.0),)),
    )
    np.testing.assert_allclose(result.state.positions[0], (1.5, 0.5, 0.5), atol=1e-12)
    np.testing.assert_allclose(result.state.geometric_lengths, (1.0,), atol=1e-12)
    np.testing.assert_allclose(result.state.optical_lengths, (1.0,), atol=1e-12)
    assert bool(result.evidence.successful)


def test_tetrahedral_field_and_curved_schlieren_retain_route_evidence():
    vertices = np.asarray(((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)), dtype=float)
    field = phx.optics.geometric.TetrahedralRefractiveIndexField(
        vertices,
        np.asarray(((0, 1, 2, 3),)),
        np.asarray((1.0, 1.02, 1.0, 1.0)),
        _contract(),
        field_id="linear-index",
    )
    n, gradient, hessian, valid = field.sample(np.asarray(((0.2, 0.2, 0.2),)))
    np.testing.assert_allclose(n, (1.002,))
    np.testing.assert_allclose(gradient, ((0.01, 0.0, 0.0),))
    np.testing.assert_allclose(hessian, 0.0)
    assert bool(valid[0])

    uniform = phx.optics.geometric.StructuredRefractiveIndexField(
        np.ones((4, 4, 4)), (0, 0, 0), (1, 1, 1), _contract(), field_id="image-index"
    )
    rays = phx.optics.geometric.GradedIndexRayPlan(uniform, 0.1, 5).prepare()
    image = phx.imaging.ImagePlaneSupport((2, 2), detector_frame_id="detector")
    quantity = phx.measurement.QuantitySpec(
        "schlieren",
        "curved-deflection",
        "deflection",
        phx.units.RADIAN,
        "sensor.deflection",
    )
    plan = phx.imaging.CurvedSchlierenPlan(
        rays,
        image,
        np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        quantity,
    )
    positions = np.asarray(
        ((0.5, 0.5, 0.5), (0.5, 1.0, 0.5), (1.0, 0.5, 0.5), (1.0, 1.0, 0.5))
    )
    directions = np.tile((0.0, 0.0, 1.0), (4, 1))
    result = plan.evaluate(positions, directions)
    np.testing.assert_allclose(result.prediction.values, 0.0, atol=1e-12)
    assert bool(result.successful)


def test_smooth_focusing_index_detects_a_caustic_without_clipping():
    field = phx.optics.geometric.AnalyticRefractiveIndexField(
        lambda point: 1.0 - 0.1 * (point[0] - 2.0) ** 2,
        _contract(),
        field_id="focusing-index",
    )
    rays = phx.optics.geometric.GradedIndexRayPlan(field, 0.02, 450).prepare()
    result = rays.integrate(
        np.asarray(((2.0, 0.0, 0.0),)),
        np.asarray(((0.0, 0.0, 1.0),)),
    )
    fan = phx.optics.geometric.RayFanPlan(
        np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    ).evaluate(result)
    assert bool(result.evidence.successful)
    assert bool(fan.caustic_detected[0])
    assert int(fan.caustic_crossings[0]) >= 1

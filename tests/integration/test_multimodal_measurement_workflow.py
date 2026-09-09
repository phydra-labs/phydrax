import numpy as np

import phydrax as phx


def test_ct_and_mri_predictions_compare_in_native_measurement_spaces():
    contract = phx.SpatialCoordinateContract(
        phx.units.METER, coordinate_system="cartesian", reference_frame="scanner"
    )
    rays = phx.measurement.RaySampleSupport(
        np.asarray(((-1.0, 0.5, 0.5),)),
        np.asarray(((1.0, 0.0, 0.0),)),
        ("ray",),
        contract,
        far=np.asarray((4.0,)),
    )
    support = phx.imaging.tomography.ProjectionSupport(rays, (1,), ("view",))
    transform = phx.imaging.tomography.VoxelXRayTransformPlan(
        support, (2, 1, 1), (0, 0, 0), (1, 1, 1)
    )
    predicted_values = transform.forward(np.ones((2, 1, 1))).values
    quantity = phx.measurement.QuantitySpec(
        "ct",
        "attenuation-line-integral",
        "attenuation-line-integral",
        phx.units.ONE,
        "ct.attenuation-line-integral",
    )
    sampling = phx.measurement.SamplingSemantics(
        phx.measurement.SpatialSamplingKind.PATH_INTEGRAL
    )
    observed = phx.measurement.QuantityField(
        "ct-observed",
        quantity,
        phx.measurement.ValueLayout.scalar(),
        support,
        sampling,
        np.asarray(predicted_values),
    ).prepare()
    predicted = phx.measurement.PreparedQuantityField(
        predicted_values,
        np.asarray((True,)),
        standard_uncertainty=None,
        quantity_id=quantity.quantity_id,
        compatibility_id=quantity.compatibility_id,
        layout_id=phx.measurement.ValueLayout.scalar().layout_id,
        support_id=support.support_id,
        sampling_id=sampling.sampling_id,
        unit_id=quantity.unit.unit_id,
        field_id="ct-predicted",
    )
    comparison = phx.observation.MeasurementComparisonPlan(observed).evaluate(predicted)
    np.testing.assert_allclose(comparison.quadratic, 0.0)
    assert bool(comparison.successful)

    coils = phx.imaging.mri.CoilSensitivityField(
        np.ones((1, 2, 2), dtype=np.complex64), ("coil",), field_id="coil"
    )
    encoding = phx.imaging.mri.CartesianMRIEncodingPlan(coils)
    image = np.asarray(((1.0 + 1.0j, 0.0), (0.0, 0.0)), dtype=np.complex64)
    kspace, _ = encoding.forward(image)
    np.testing.assert_allclose(encoding.adjoint(kspace), image, atol=1e-6)

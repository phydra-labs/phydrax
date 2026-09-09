import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _manifest():
    return phx.qualification.ReferenceArtifactManifest(
        "synthetic-schlieren",
        checksum_algorithm="sha256",
        checksum="4" * 64,
        size_bytes=1,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"signal": 0.0},
        lineage_ids=("synthetic",),
    )


def _background(support):
    quantity = phx.measurement.QuantitySpec(
        "imaging", "detector-signal", "detector-signal", phx.units.ONE, "sensor.signal"
    )
    field = phx.measurement.QuantityField(
        "background",
        quantity,
        phx.measurement.ValueLayout.scalar(),
        support,
        phx.measurement.SamplingSemantics(
            phx.measurement.SpatialSamplingKind.DETECTOR_BIN
        ),
        np.asarray(((1.0, 2.0), (3.0, 4.0))),
    )
    asset = phx.measurement.MeasurementAsset.from_single_reference(
        "background",
        field,
        _manifest(),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RAW,
            transformation_id="background-generator",
        ),
    )
    return phx.imaging.ImageAsset(asset)


def test_schlieren_deflection_and_image_formation_keep_quantities_distinct():
    spatial = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-world",
        reference_frame="world",
    )
    image = phx.imaging.ImagePlaneSupport((2, 2), detector_frame_id="detector")
    rays = phx.measurement.RaySampleSupport(
        np.zeros((4, 3)),
        np.tile((0.0, 0.0, 1.0), (4, 1)),
        tuple(f"pixel-{index}" for index in range(4)),
        spatial,
    )
    density_gradient_unit = phx.units.derived_unit(
        "kg/m4", ((phx.units.KILOGRAM, 1), (phx.units.METER, -4))
    )
    coefficient_unit = phx.units.derived_unit(
        "m3/kg", ((phx.units.METER, 3), (phx.units.KILOGRAM, -1))
    )
    gradient_quantity = phx.measurement.QuantitySpec(
        "schlieren",
        "density-gradient",
        "density-gradient",
        density_gradient_unit,
        "physical.density-gradient",
    )
    deflection_quantity = phx.measurement.QuantitySpec(
        "schlieren",
        "angular-deflection",
        "angular-deflection",
        phx.units.RADIAN,
        "sensor.angular-deflection",
    )
    plan = phx.imaging.SchlierenDeflectionPlan(
        rays,
        image,
        np.full((4, 2), 0.5),
        np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        gradient_quantity,
        deflection_quantity,
        relation=phx.imaging.GladstoneDaleRelation(
            0.1, coefficient_unit, "synthetic-medium"
        ),
        gradient_frame_id="world",
        small_angle_limit=0.5,
    )
    gradients = np.zeros((4, 2, 3))
    gradients[..., 0] = 2.0
    result = plan.evaluate(gradients)
    compiled = eqx.filter_jit(plan.evaluate)(gradients)
    np.testing.assert_allclose(compiled.deflection_rc, result.deflection_rc)
    np.testing.assert_allclose(result.deflection_rc[..., 0], 0.2)
    np.testing.assert_allclose(result.deflection_rc[..., 1], 0.0)
    assert bool(result.successful & result.small_angle_valid)
    derivative = jax.grad(
        lambda scale: jnp.sum(plan.evaluate(gradients * scale).deflection_rc)
    )(jnp.asarray(1.0))
    np.testing.assert_allclose(derivative, 0.8)
    millimeter_spatial = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-world",
        reference_frame="world",
    )
    millimeter_rays = phx.measurement.RaySampleSupport(
        np.zeros((4, 3)),
        np.tile((0.0, 0.0, 1.0), (4, 1)),
        tuple(f"millimeter-pixel-{index}" for index in range(4)),
        millimeter_spatial,
    )
    scaled = phx.imaging.SchlierenDeflectionPlan(
        millimeter_rays,
        image,
        np.full((4, 2), 500.0),
        np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        gradient_quantity,
        deflection_quantity,
        relation=phx.imaging.GladstoneDaleRelation(
            0.1, coefficient_unit, "synthetic-medium"
        ),
        gradient_frame_id="world",
        small_angle_limit=0.5,
    ).evaluate(gradients)
    np.testing.assert_allclose(scaled.deflection_rc, result.deflection_rc)
    background = _background(image)
    pair = phx.imaging.SchlierenImagePair(
        background,
        background,
        phx.imaging.SchlierenMethod.KNIFE_EDGE,
        "synthetic-calibration",
    )
    assert pair.reference.image_id == pair.observed.image_id
    knife = phx.imaging.KnifeEdgeSchlierenPlan(
        background, (1.0, 0.0), contrast_gain_per_radian=2.0
    ).evaluate(result)
    compiled_knife = eqx.filter_jit(
        phx.imaging.KnifeEdgeSchlierenPlan(
            background, (1.0, 0.0), contrast_gain_per_radian=2.0
        ).evaluate
    )(result)
    np.testing.assert_allclose(compiled_knife.prediction.values, knife.prediction.values)
    np.testing.assert_allclose(knife.prediction.values, background.field.values * 1.4)
    assert knife.prediction.compatibility_id != result.prediction.compatibility_id
    bos = phx.imaging.BackgroundOrientedSchlierenPlan(background, 1.0).evaluate(result)
    compiled_bos = eqx.filter_jit(
        phx.imaging.BackgroundOrientedSchlierenPlan(background, 1.0).evaluate
    )(result)
    np.testing.assert_allclose(compiled_bos.prediction.values, bos.prediction.values)
    assert bos.displacement_rc.shape == (2, 2, 2)
    assert bool(bos.successful)

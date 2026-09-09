import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _surface():
    vertices = np.asarray(
        (
            (-1.0, -1.0, 2.0),
            (1.0, -1.0, 2.0),
            (0.0, 1.0, 2.0),
            (0.0, 0.0, 4.0),
        )
    )
    triangles = np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)))
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-world",
        reference_frame="world",
    )
    metadata = phx.geometry.surface.SurfaceMetadata(
        source_id="tetrahedron",
        source_revision="1",
        coordinate_contract=contract,
        provenance=("synthetic",),
    )
    model = phx.geometry.surface.SurfaceModel.from_triangles(
        vertices,
        triangles,
        metadata,
        repair_orientation=True,
        orient_closed_outward=True,
    )
    return model.prepare(), vertices


def test_surface_image_refits_dynamic_geometry_and_interpolates_vertex_fields():
    realization, vertices = _surface()
    support = phx.imaging.ImagePlaneSupport((3, 3), detector_frame_id="camera")
    camera = phx.imaging.camera.CameraModel(
        phx.imaging.camera.CameraIntrinsics((1.0, 1.0), (1.0, 1.0), image_shape=(3, 3))
    )
    quantity = phx.measurement.QuantitySpec(
        "rendering", "surface-field", "surface-field", phx.units.ONE, "test.surface-field"
    )
    prepared = phx.rendering.SurfaceImagePlan(
        realization,
        support,
        camera,
        quantity,
        phx.measurement.ValueLayout.scalar(),
        phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.POINT),
    ).prepare()
    values = jnp.asarray((2.0, 2.0, 2.0, 4.0))
    base = prepared.render(vertices, values, geometry_id="base")
    assert bool(base.hit[1, 1])
    np.testing.assert_allclose(base.depth[1, 1], 2.0)
    np.testing.assert_allclose(base.prediction.values[1, 1], 2.0)
    compiled = eqx.filter_jit(prepared.render)(
        jnp.asarray(vertices), values, geometry_id="compiled-surface"
    )
    np.testing.assert_allclose(compiled.prediction.values, base.prediction.values)
    moved_vertices = vertices + np.asarray((0.0, 0.0, 1.0))
    moved = prepared.render(moved_vertices, values + 1.0, geometry_id="moved")
    np.testing.assert_allclose(moved.depth[1, 1], 3.0)
    np.testing.assert_allclose(moved.prediction.values[1, 1], 3.0)
    assert bool(moved.evidence.successful)
    derivative = jax.grad(
        lambda offset: prepared.render(
            jnp.asarray(vertices) + jnp.asarray((0.0, 0.0, offset)),
            values,
            geometry_id="differentiable-motion",
        ).depth[1, 1]
    )(jnp.asarray(0.0))
    np.testing.assert_allclose(derivative, 1.0)
    field_gradient = jax.grad(
        lambda current: prepared.render(
            vertices,
            current,
            geometry_id="differentiable-field",
        ).prediction.values[1, 1]
    )(values)
    np.testing.assert_allclose(jnp.sum(field_gradient), 1.0)


def test_lidar_surface_prediction_reuses_exact_dynamic_visibility():
    realization, vertices = _surface()
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-world",
        reference_frame="world",
    )
    rays = phx.measurement.RaySampleSupport(
        np.zeros((1, 3)),
        np.asarray(((0.0, 0.0, 1.0),)),
        ("pulse-0",),
        contract,
        far=np.asarray((10.0,)),
    )
    quantity = phx.measurement.QuantitySpec(
        "lidar", "range", "range", phx.units.METER, "physical.range"
    )
    prepared = phx.rendering.LidarSurfacePlan(
        realization,
        rays,
        quantity,
        phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.EVENT),
    ).prepare()
    result = prepared.predict(vertices, geometry_id="base-lidar")
    np.testing.assert_allclose(result.prediction.values, (2.0,))
    assert bool(result.prediction.valid_mask[0] & result.evidence.successful)
    compiled = eqx.filter_jit(prepared.predict)(
        jnp.asarray(vertices), geometry_id="compiled-lidar"
    )
    np.testing.assert_allclose(compiled.prediction.values, (2.0,))
    moved = prepared.predict(
        vertices + np.asarray((0.0, 0.0, 1.0)),
        geometry_id="moved-lidar",
    )
    np.testing.assert_allclose(moved.prediction.values, (3.0,))

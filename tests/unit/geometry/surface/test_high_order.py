import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry import (
    HighOrderDifferentiationError,
    realize_isoparametric_triangles,
    SurfaceMetadata,
    SurfaceModel,
)


def test_linear_isoparametric_triangle_preserves_authoritative_geometry_and_frame():
    metadata = SurfaceMetadata(
        source_id="unit-triangle",
        source_revision="1",
        length_unit="m",
        provenance=("native-test",),
        cell_tags=("face",),
    )
    model = SurfaceModel.from_triangles(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([[0, 1, 2]], dtype=np.int32),
        metadata,
    )
    nodes = np.asarray([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    realized = realize_isoparametric_triangles(model, nodes, 1)
    frame = realized.frame(jnp.asarray([0]), jnp.asarray([[1.0 / 3.0, 1.0 / 3.0]]))

    assert jnp.allclose(
        frame.physical_coordinates, jnp.asarray([[1.0 / 3.0, 1.0 / 3.0, 0.0]])
    )
    assert jnp.allclose(frame.jacobian, jnp.ones((1,)))
    assert jnp.allclose(frame.normal, jnp.asarray([[0.0, 0.0, 1.0]]))
    assert bool(frame.valid[0])
    assert frame.metric_unit == "m^2"
    assert frame.jacobian_unit == "m^2"

    with pytest.raises(HighOrderDifferentiationError):
        realized.evaluate(
            jnp.asarray([0]),
            jnp.asarray([[0.2, 0.3]]),
            derivative_order=2,
        )

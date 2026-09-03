import jax.numpy as jnp

from phydrax.geometry import box_closest_point, radial_closest_point


def test_radial_closest_point_carries_physical_exactness_and_signed_coordinate():
    points = jnp.asarray([[2.0, 0.0], [0.0, 0.5]])
    result = radial_closest_point(
        points,
        jnp.zeros((2,)),
        jnp.asarray(1.0),
        represented_geometry_id="unit-circle",
    )
    assert jnp.allclose(result.closest_point, jnp.asarray([[1.0, 0.0], [0.0, 1.0]]))
    assert jnp.allclose(result.normal_coordinate, jnp.asarray([1.0, -0.5]))
    assert jnp.all(result.unique)
    assert result.exact_to_physical


def test_box_closest_point_marks_face_interior_regular_and_corner_tie_nonregular():
    result = box_closest_point(
        jnp.asarray([[2.0, 0.0], [2.0, 2.0]]),
        jnp.zeros((2,)),
        jnp.asarray([2.0, 2.0]),
        represented_geometry_id="unit-box",
    )
    assert jnp.allclose(result.closest_point[0], jnp.asarray([1.0, 0.0]))
    assert result.regular[0]
    assert not result.regular[1]

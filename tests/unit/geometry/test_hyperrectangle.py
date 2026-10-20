#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def test_hyperrectangle_contracts() -> None:
    geometry = phx.domain.HyperRectangle(
        lower=jnp.asarray([-1.0, 0.0, 2.0]),
        upper=jnp.asarray([1.0, 3.0, 6.0]),
    )
    assert geometry.label == "x"
    assert geometry.spatial_dim == 3
    assert np.allclose(
        np.asarray(geometry.bounds),
        [[-1.0, 0.0, 2.0], [1.0, 3.0, 6.0]],
    )
    assert np.isclose(float(geometry.volume), 24.0)
    assert np.isclose(float(geometry.boundary_measure_value), 52.0)

    planar = phx.domain.HyperRectangle(
        lower=jnp.asarray([0.0, -1.0]),
        upper=jnp.asarray([2.0, 3.0]),
    )
    points = jnp.asarray([[1.0, 0.0], [0.0, 0.0], [2.0, 3.0], [3.0, 0.0]])
    np.testing.assert_array_equal(planar._contains(points), [True, True, True, False])
    np.testing.assert_array_equal(
        planar._on_boundary(points),
        [False, True, True, False],
    )
    expected_normals = jnp.asarray(
        [[-1.0, 0.0], [1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)]]
    )
    np.testing.assert_allclose(
        planar._boundary_normals(jnp.asarray([[0.0, 0.0], [2.0, 3.0]])),
        expected_normals,
    )
    signed_distance = planar.adf(points)
    assert signed_distance[0] < 0.0
    assert np.isclose(float(signed_distance[1]), 0.0)
    assert np.isclose(float(signed_distance[2]), 0.0)
    assert signed_distance[3] > 0.0

    with pytest.raises(ValueError, match="matching shapes"):
        phx.domain.HyperRectangle(lower=jnp.zeros((2,)), upper=jnp.ones((3,)))
    with pytest.raises(ValueError, match="upper > lower"):
        phx.domain.HyperRectangle(
            lower=jnp.asarray([0.0, 1.0]),
            upper=jnp.asarray([1.0, 1.0]),
        )
    geometry = phx.domain.HyperRectangle(
        lower=jnp.asarray([-1.0, 0.0]),
        upper=jnp.asarray([1.0, 2.0]),
    )
    interior = geometry.sample_interior(16, key=jr.key(0))
    boundary = geometry.sample_boundary(16, key=jr.key(1))
    assert interior.shape == boundary.shape == (16, 2)
    assert bool(jnp.all(geometry._contains(interior)))
    assert bool(jnp.all(geometry._on_boundary(boundary)))
    np.testing.assert_allclose(
        jnp.linalg.norm(geometry._boundary_normals(boundary), axis=-1),
        1.0,
    )

    unfiltered = geometry.sample_interior(4, sampler="hammersley", key=jr.key(2))
    assert unfiltered.shape == (4, 2)
    for sample in (geometry.sample_interior, geometry.sample_boundary):
        with pytest.raises(ValueError, match="prefix-stable or randomized"):
            sample(
                4,
                where=lambda point: point[0] < 0.5,
                sampler="hammersley",
                key=jr.key(3),
            )

    grid_geometry = phx.domain.HyperRectangle(
        lower=jnp.asarray([0.0, 1.0]),
        upper=jnp.asarray([2.0, 3.0]),
    )
    grid = grid_geometry.component().sample(
        phx.domain.GridSampling(
            {
                "x": (
                    phx.discretization.UniformAxisSpec(5),
                    phx.discretization.UniformAxisSpec(7),
                )
            }
        ),
        key=jr.key(0),
    )
    x0, x1 = grid["x"]
    assert x0.data.shape == (5,)
    assert x1.data.shape == (7,)
    assert grid.coord_mask_by_label["x"].data.shape == (5, 7)
    geom = phx.domain.HyperRectangle(
        lower=jnp.array([-1.0, 0.0]),
        upper=jnp.array([1.0, 2.0]),
    )
    points = jnp.array([[0.75, 0.25], [-0.5, 1.5]])
    displacement = jnp.array([[4.5, -3.0], [-5.0, 6.0]])
    transition = eqx.filter_jit(geom.transition_interior)
    result = transition(points, displacement)

    assert bool(jnp.all(result.valid))
    assert bool(jnp.all(geom._contains(result.points)))
    assert bool(jnp.all(result.reflection_count > 0))
    geom = phx.domain.HyperRectangle(lower=jnp.zeros((6,)), upper=jnp.ones((6,)))
    model = phx.nn.models.SeparableMLP(
        in_size=6,
        out_size="scalar",
        latent_size=4,
        width_size=8,
        depth=1,
        key=jr.key(0),
    )
    u = geom.Model("x")(model)

    batch = geom.component().sample(
        phx.domain.PointSampling(3, layout=phx.domain.SampleLayout((("x",),))),
        key=jr.key(0),
    )
    out = u(batch)
    axis = batch.structure.axis_for("x")
    assert out.dims == (axis,)
    assert out.data.shape == (3,)


def test_hyperrectangle_finite_observation_with_stacked_points() -> None:
    geom = phx.domain.HyperRectangle(lower=jnp.zeros((2,)), upper=jnp.ones((2,)))

    @geom.Function("x")
    def exact(x: Any) -> Any:
        return x[0] + 2.0 * x[1]

    @geom.Function("x")
    def u(x: Any) -> Any:
        return x[0] + 2.0 * x[1]

    points = jnp.array([[0.1, 0.2], [0.4, 0.5], [0.8, 0.3]], dtype="float64")
    component = geom.component()
    batch = component.points(points)
    condition = phx.conditions.Observation("u", component, exact)
    source = phx.integration.fixed(
        phx.integration.from_samples(phx.integration.mean_over(component), batch)
    )
    term = phx.terms.ObservationPenalty(condition, source)

    loss = term.loss({"u": u}, key=jr.key(0))
    assert loss < 1e-10


def test_hyperrectangle_explicit_float32_coordinates() -> None:
    lower, upper = np.asarray([0.0, -1.0]), np.asarray([2.0, 3.0])
    default = phx.domain.HyperRectangle(lower, upper)
    box = phx.domain.HyperRectangle(lower, upper, dtype=jnp.float32)
    line = phx.domain.HyperRectangle(np.zeros(1), np.full(1, 2.0), dtype="float32")

    assert default.lower.dtype == default.upper.dtype == jnp.float64
    points = jnp.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=jnp.float32)
    returned = {
        "lower": box.lower,
        "upper": box.upper,
        "bounds": box.bounds,
        "volume": box.volume,
        "boundary measure": box.boundary_measure_value,
        "line boundary measure": line.boundary_measure_value,
        "adf": box.adf(points),
        "normals": box._boundary_normals(points),
        "interior": box.sample_interior(8, key=jr.key(0)),
        "boundary": box.sample_boundary(8, key=jr.key(1)),
        "filtered": box.sample_interior(
            8, where=lambda x: x[0] > 1.0, sampler="uniform", key=jr.key(2)
        ),
        "separable": box._sample_interior_separable(3, key=jr.key(3))[0][1],
    }
    for role, value in returned.items():
        assert value.dtype == jnp.float32, role
    np.testing.assert_allclose(
        returned["normals"], [[-1.0, 0.0], [2.0**-0.5, 2.0**-0.5]], rtol=1e-6
    )
    assert bool(jnp.all(box._contains(returned["interior"])))
    assert bool(jnp.all(box._on_boundary(returned["boundary"])))
    assert bool(jnp.all(returned["filtered"][:, 0] > 1.0))


@pytest.mark.parametrize(
    ("dtype", "message"),
    (
        (jnp.float16, "support float32 or float64"),
        (jnp.int32, "'int32'"),
        (jnp.complex64, "not real floating-point"),
    ),
)
def test_hyperrectangle_refuses_unsupported_coordinate_dtype_first(
    dtype: Any, message: str
) -> None:
    # Mismatched bounds would fail later; the dtype refusal comes first.
    with pytest.raises(ValueError, match=message):
        phx.domain.HyperRectangle(jnp.zeros((2,)), jnp.ones((3,)), dtype=dtype)

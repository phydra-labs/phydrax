#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def test_hyperrectangle_basic_measures():
    geom = phx.domain.HyperRectangle(
        lower=jnp.array([-1.0, 0.0, 2.0]),
        upper=jnp.array([1.0, 3.0, 6.0]),
    )

    assert geom.label == "x"
    assert geom.spatial_dim == 3
    assert np.allclose(np.asarray(geom.bounds), [[-1.0, 0.0, 2.0], [1.0, 3.0, 6.0]])
    assert np.isclose(float(geom.volume), 24.0)

    # Surface area of a 2 x 3 x 4 cuboid: 2 * (3*4 + 2*4 + 2*3).
    assert np.isclose(float(geom.boundary_measure_value), 52.0)


def test_hyperrectangle_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="matching shapes"):
        phx.domain.HyperRectangle(lower=jnp.zeros((2,)), upper=jnp.ones((3,)))

    with pytest.raises(ValueError, match="upper > lower"):
        phx.domain.HyperRectangle(
            lower=jnp.array([0.0, 1.0]), upper=jnp.array([1.0, 1.0])
        )


def test_hyperrectangle_contains_boundary_normals_and_adf():
    geom = phx.domain.HyperRectangle(
        lower=jnp.array([0.0, -1.0]), upper=jnp.array([2.0, 3.0])
    )
    pts = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [2.0, 3.0],
            [3.0, 0.0],
        ]
    )

    assert np.allclose(np.asarray(geom._contains(pts)), [True, True, True, False])
    assert np.allclose(np.asarray(geom._on_boundary(pts)), [False, True, True, False])

    normals = geom._boundary_normals(jnp.array([[0.0, 0.0], [2.0, 3.0]]))
    expected = jnp.array([[-1.0, 0.0], [1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)]])
    assert np.allclose(np.asarray(normals), np.asarray(expected))

    sdf = geom.adf(pts)
    assert sdf[0] < 0.0
    assert np.isclose(float(sdf[1]), 0.0)
    assert np.isclose(float(sdf[2]), 0.0)
    assert sdf[3] > 0.0


def test_hyperrectangle_sampling_shapes_and_membership():
    geom = phx.domain.HyperRectangle(
        lower=jnp.array([-1.0, 0.0]), upper=jnp.array([1.0, 2.0])
    )

    interior = geom.sample_interior(16, key=jr.key(0))
    assert interior.shape == (16, 2)
    assert bool(jnp.all(geom._contains(interior)))

    boundary = geom.sample_boundary(16, key=jr.key(1))
    assert boundary.shape == (16, 2)
    assert bool(jnp.all(geom._on_boundary(boundary)))
    normals = geom._boundary_normals(boundary)
    assert np.allclose(np.asarray(jnp.linalg.norm(normals, axis=-1)), 1.0)


def test_hyperrectangle_reflects_large_adaptive_moves_under_jit():
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


def test_hyperrectangle_coord_separable_sampling():
    geom = phx.domain.HyperRectangle(
        lower=jnp.array([0.0, 1.0]), upper=jnp.array([2.0, 3.0])
    )
    batch = geom.component().sample(
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

    x0, x1 = batch["x"]
    assert x0.data.shape == (5,)
    assert x1.data.shape == (7,)
    assert batch.coord_mask_by_label["x"].data.shape == (5, 7)


def test_hyperrectangle_finite_observation_with_stacked_points():
    geom = phx.domain.HyperRectangle(lower=jnp.zeros((2,)), upper=jnp.ones((2,)))

    @geom.Function("x")
    def exact(x):
        return x[0] + 2.0 * x[1]

    @geom.Function("x")
    def u(x):
        return x[0] + 2.0 * x[1]

    points = jnp.array([[0.1, 0.2], [0.4, 0.5], [0.8, 0.3]], dtype=float)
    component = geom.component()
    batch = component.points(points)
    condition = phx.conditions.Observation("u", component, exact)
    source = phx.integration.fixed(
        phx.integration.from_samples(phx.integration.mean_over(component), batch)
    )
    term = phx.terms.ObservationPenalty(condition, source)

    loss = term.loss({"u": u}, key=jr.key(0))
    assert loss < 1e-10


def test_hyperrectangle_domain_model_gets_vector_points():
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

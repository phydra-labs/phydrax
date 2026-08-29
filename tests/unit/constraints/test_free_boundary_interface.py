#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax._frozendict import frozendict


def test_interface_feature_lift_encodes_normalized_distance_cusp_and_side():
    lift = phx.nn.layers.InterfaceFeatureLift(
        lambda point: 3.0 * point[0],
        2,
        distance_semantics="level_set",
        distance_clip=1.0,
        side_width=0.2,
    )

    features = lift(jnp.asarray((0.1, 0.5)))

    assert lift.in_size == 2
    assert lift.out_size == 5
    np.testing.assert_allclose(features[:2], (0.1, 0.5), atol=1.0e-14)
    np.testing.assert_allclose(features[2], 0.1, atol=1.0e-14)
    np.testing.assert_allclose(features[3], 0.1, atol=1.0e-14)
    assert 0.5 < features[4] < 1.0


def test_exact_stefan_interface_condition_factories_have_zero_residual():
    spatial = phx.domain.GeometryDomain(
        phx.geometry.Rectangle(center=(0.75, 0.0), size=(1.5, 1.0)).compile()
    )
    domain = spatial @ phx.domain.TimeInterval(0.0, 0.5)
    component = domain.component()

    @domain.Function("x", "t")
    def inside(point, time):
        return jnp.exp(time + 0.5 - point[0]) - 1.0

    @domain.Function("x", "t")
    def outside(point, time):
        del point, time
        return 0.0

    @domain.Function("x", "t")
    def level_set(point, time):
        return point[0] - time - 0.5

    condition = phx.conditions.free_boundary.StefanBalance(
        "inside",
        "outside",
        "phi",
        component,
        inside_conductivity=1.0,
        outside_conductivity=0.0,
        volumetric_latent_heat=1.0,
    )
    residual = condition.residual(
        {"inside": inside, "outside": outside, "phi": level_set}
    )
    times = jnp.asarray((0.0, 0.2, 0.4))
    points = jnp.stack((times + 0.5, jnp.zeros_like(times)), axis=-1)
    batch = frozendict(
        {
            "x": cx.Field(points, dims=("point", None)),
            "t": cx.Field(times, dims=("point",)),
        }
    )

    np.testing.assert_allclose(residual(batch).data, 0.0, atol=1.0e-12)


def test_causal_schedule_and_narrow_band_policy_retain_interface_points():
    schedule = phx.sampling.collocation.CausalTimeSlabSchedule(
        (0.0, 0.5, 1.0),
        overlap_fraction=0.2,
        causal_strength=2.0,
    )
    weights = schedule.causal_weights(jnp.asarray((0.25, 4.0)))
    np.testing.assert_allclose(weights, (1.0, jnp.exp(-0.5)), atol=1.0e-14)
    assert bool(schedule.active(jnp.asarray(0.45), 1))

    domain = phx.domain.Interval1d(-1.0, 1.0)

    @domain.Function("x")
    def level_set(point):
        return point[0]

    @domain.Function("x")
    def residual_coordinate(point):
        return 0.01 * point[0] ** 2

    base = phx.sampling.collocation.R3(
        refresh_every=1,
        sampler="uniform",
        max_retain_fraction=0.5,
    )
    policy = phx.sampling.collocation.NarrowBandCollocationPolicy(
        "phi",
        0.2,
        base_policy=base,
        band_strength=10.0,
    )
    condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda _u: residual_coordinate,
    )
    source = phx.integration.adaptive(
        phx.integration.mean_over(condition.on),
        phx.domain.PointSampling(
            64,
            layout=phx.domain.SampleLayout((("x",),)),
            design="uniform",
        ),
        policy,
    )
    term = phx.terms.ResidualPenalty(condition, source)
    functions = {"u": domain.Function()(0.0), "phi": level_set}
    initial = policy.initialize(term, key=jr.key(0))
    refreshed = policy.refresh(
        term,
        functions,
        initial,
        key=jr.key(1),
        iter_=1,
    )
    coordinates = jnp.asarray(refreshed.batch.points["x"].data).reshape((-1,))

    assert coordinates.shape == (64,)
    assert jnp.sum(jnp.abs(coordinates) < 0.2) >= 8

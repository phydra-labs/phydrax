#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.random as jr

import phydrax as phx
from phydrax.domain import Interval1d, SampleLayout, TimeInterval


def _xt_domain():
    return Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)


def test_sensor_track_observation_matches_linear_time_target():
    domain = _xt_domain()
    structure = SampleLayout((("x", "t"),))

    @domain.Function("x", "t")
    def u(x, t):
        return 2.0 * t

    component = domain.component()
    target = domain.Function("x", "t")(lambda x, t: 2.0 * t)
    condition = phx.conditions.Observation("u", component, target)
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(16, layout=structure),
    )
    term = phx.terms.ObservationPenalty(condition, source)
    loss_fn = eqx.filter_jit(lambda k: term.loss({"u": u}, key=k))
    assert loss_fn(jr.key(0)) < 1e-6


def test_sensor_tracks_single_time_constant():
    domain = _xt_domain()
    structure = SampleLayout((("x", "t"),))

    @domain.Function("x", "t")
    def u(x, t):
        return 3.0

    component = domain.component()
    target = domain.Function()(3.0)
    condition = phx.conditions.Observation("u", component, target)
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(16, layout=structure),
    )
    term = phx.terms.ObservationPenalty(condition, source)
    loss_fn = eqx.filter_jit(lambda k: term.loss({"u": u}, key=k))
    assert loss_fn(jr.key(0)) < 1e-6

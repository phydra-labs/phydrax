#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.random as jr
import pytest

import phydrax as phx
from phydrax.conditions import Initial
from phydrax.domain import (
    Boundary,
    ComponentSum,
    GridBatch,
    Interior,
    Interval1d,
    SampleLayout,
    TimeInterval,
)


def test_grid_sampling_rejects_component_sum():
    geom = Interval1d(0.0, 1.0)
    c1 = geom.component({"x": Interior()})
    c2 = geom.component({"x": Boundary()})
    union = ComponentSum((c1, c2))

    with pytest.raises(TypeError, match="does not support GridSampling"):
        union.sample(
            phx.domain.GridSampling({"x": 4}),
            key=jr.key(0),
        )


def test_grid_sampling_requires_dense_plan_for_other_labels():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component()

    with pytest.raises(ValueError, match="GridSampling.dense is required"):
        component.sample(phx.domain.GridSampling({"x": 4}), key=jr.key(0))


def test_grid_sampling_accepts_scalar_labels():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component()
    sampling = phx.domain.GridSampling(
        {"t": 4},
        dense=phx.domain.PointSampling(
            5,
            layout=SampleLayout((("x",),)),
        ),
    )

    batch = component.sample(sampling, key=jr.key(0))
    assert isinstance(batch, GridBatch)
    assert "t" in batch.coord_axes_by_label
    assert len(batch.points["t"]) == 1


def test_continuous_initial_requires_fixed_start():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"t": Interior()})

    with pytest.raises(ValueError, match="FixedStart"):
        Initial("u", component, target=0.0)


def test_domain_join_rejects_label_collision():
    left = Interval1d(0.0, 1.0)
    right = Interval1d(-1.0, 2.0)
    with pytest.raises(ValueError, match="Label collision"):
        _ = left @ right

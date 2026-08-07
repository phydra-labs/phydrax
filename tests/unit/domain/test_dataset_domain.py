#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.domain import (
    DATASET_INDEX_KEY,
    DatasetDomain,
    FourierAxisSpec,
    Interval1d,
    SampleLayout,
)
from phydrax.integration import from_samples, over
from phydrax.operators.integral import integral


def test_dataset_domain_samples_points_batch():
    data = jnp.arange(10.0, dtype=float).reshape((10, 1))
    dom = DatasetDomain(data)
    component = dom.component()
    structure = SampleLayout((("data",),))

    batch = component.sample(phx.domain.PointSampling(4, layout=structure), key=jr.key(0))
    axis = batch.structure.axis_for("data")
    assert axis is not None

    field = batch["data"]
    assert field.dims == (axis, None)
    assert field.data.shape == (4, 1)


def test_dataset_domain_points_from_indices_carries_internal_indices():
    data = jnp.arange(10.0, dtype=float).reshape((5, 2))
    dom = DatasetDomain(data)
    structure = SampleLayout((("data",),))
    indices = jnp.asarray([3, 1, 3], dtype=jnp.int32)

    batch = dom.points_from_indices(indices, structure=structure)
    axis = batch.structure.axis_for("data")
    assert axis is not None
    assert batch["data"].dims == (axis, None)
    assert jnp.allclose(batch["data"].data, data[indices])
    assert jnp.all(batch[DATASET_INDEX_KEY].data == indices)


def test_dataset_domain_integral_probability_measure_is_average():
    data = jnp.zeros((5, 2), dtype=float)
    dom = DatasetDomain(data, measure="probability")
    component = dom.component()
    structure = SampleLayout((("data",),))

    batch = component.sample(phx.domain.PointSampling(3, layout=structure), key=jr.key(0))
    u = dom.Function()(1.0)
    realization = from_samples(over(component), batch)
    out = integral(u, realization)
    assert jnp.allclose(jnp.asarray(out.data), 1.0)


def test_dataset_domain_integral_count_measure_is_sum():
    data = jnp.zeros((5, 2), dtype=float)
    dom = DatasetDomain(data, measure="count")
    component = dom.component()
    structure = SampleLayout((("data",),))

    batch = component.sample(phx.domain.PointSampling(3, layout=structure), key=jr.key(0))
    u = dom.Function()(1.0)
    realization = from_samples(over(component), batch)
    out = integral(u, realization)
    assert jnp.allclose(jnp.asarray(out.data), 5.0)


def test_dataset_domain_with_coord_separable_geometry_sampling():
    data = jnp.arange(6.0, dtype=float)
    data_dom = DatasetDomain(data)
    geom = Interval1d(0.0, 1.0)
    domain = data_dom @ geom

    component = domain.component()
    dense_structure = SampleLayout((("data",),))
    batch = component.sample(
        phx.domain.GridSampling(
            {"x": FourierAxisSpec(8)},
            dense=phx.domain.PointSampling(3, layout=dense_structure),
        ),
        key=jr.key(0),
    )

    axis = batch.dense_structure.axis_for("data")
    assert axis is not None
    assert batch["data"].dims == (axis,)
    assert batch["data"].data.shape == (3,)
    assert isinstance(batch["x"], tuple)
    assert batch["x"][0].data.shape == (8,)

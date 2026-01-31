#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.domain import DatasetDomain, FourierAxisSpec, Interval1d, ProductStructure
from phydrax.operators.integral import integral


def test_dataset_domain_samples_points_batch():
    data = jnp.arange(10.0, dtype=float).reshape((10, 1))
    dom = DatasetDomain(data)
    component = dom.component()
    structure = ProductStructure((("data",),))

    batch = component.sample(4, structure=structure, key=jr.key(0))
    axis = batch.structure.axis_for("data")
    assert axis is not None

    field = batch["data"]
    assert field.dims == (axis, None)
    assert field.data.shape == (4, 1)


def test_dataset_domain_integral_probability_measure_is_average():
    data = jnp.zeros((5, 2), dtype=float)
    dom = DatasetDomain(data, measure="probability")
    component = dom.component()
    structure = ProductStructure((("data",),))

    batch = component.sample(3, structure=structure, key=jr.key(0))
    u = dom.Function()(1.0)
    out = integral(u, batch, component=component)
    assert jnp.allclose(jnp.asarray(out.data), 1.0)


def test_dataset_domain_integral_count_measure_is_sum():
    data = jnp.zeros((5, 2), dtype=float)
    dom = DatasetDomain(data, measure="count")
    component = dom.component()
    structure = ProductStructure((("data",),))

    batch = component.sample(3, structure=structure, key=jr.key(0))
    u = dom.Function()(1.0)
    out = integral(u, batch, component=component)
    assert jnp.allclose(jnp.asarray(out.data), 5.0)


def test_dataset_domain_with_coord_separable_geometry_sampling():
    data = jnp.arange(6.0, dtype=float)
    data_dom = DatasetDomain(data)
    geom = Interval1d(0.0, 1.0)
    domain = data_dom @ geom

    component = domain.component()
    dense_structure = ProductStructure((("data",),))
    batch = component.sample_coord_separable(
        {"x": FourierAxisSpec(8)},
        num_points=3,
        dense_structure=dense_structure,
        key=jr.key(0),
    )

    axis = batch.dense_structure.axis_for("data")
    assert axis is not None
    assert batch["data"].dims == (axis,)
    assert batch["data"].data.shape == (3,)
    assert isinstance(batch["x"], tuple)
    assert batch["x"][0].data.shape == (8,)

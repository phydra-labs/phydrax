#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.discretization import FourierAxisSpec
from phydrax.domain import (
    DATASET_INDEX_KEY,
    DatasetDomain,
    Interval1d,
    SampleLayout,
)
from phydrax.integration import from_samples, over
from phydrax.operators.integral import integral


def test_dataset_domain_contracts() -> None:
    data = jnp.arange(10.0, dtype="float64").reshape((10, 1))
    dom = DatasetDomain(data)
    component = dom.component()
    structure = SampleLayout((("data",),))

    batch = component.sample(phx.domain.PointSampling(4, layout=structure), key=jr.key(0))
    axis = batch.structure.axis_for("data")
    assert axis is not None

    field = batch["data"]
    assert field.dims == (axis, None)
    assert field.data.shape == (4, 1)
    data = jnp.arange(10.0, dtype="float64").reshape((5, 2))
    dom = DatasetDomain(data)
    structure = SampleLayout((("data",),))
    indices = jnp.asarray([3, 1, 3], dtype=jnp.int32)

    batch = dom.points_from_indices(indices, structure=structure)
    axis = batch.structure.axis_for("data")
    assert axis is not None
    assert batch["data"].dims == (axis, None)
    assert jnp.allclose(jnp.asarray(batch["data"].data), data[indices])
    assert jnp.all(batch[DATASET_INDEX_KEY].data == indices)
    data = jnp.zeros((5, 2), dtype="float64")
    dom = DatasetDomain(data, measure="probability")
    component = dom.component()
    structure = SampleLayout((("data",),))

    batch = component.sample(phx.domain.PointSampling(3, layout=structure), key=jr.key(0))
    u = dom.Function()(1.0)
    realization = from_samples(over(component), batch)
    out = integral(u, realization)
    assert jnp.allclose(jnp.asarray(out.data), 1.0)
    data = jnp.zeros((5, 2), dtype="float64")
    dom = DatasetDomain(data, measure="count")
    component = dom.component()
    structure = SampleLayout((("data",),))

    batch = component.sample(phx.domain.PointSampling(3, layout=structure), key=jr.key(0))
    u = dom.Function()(1.0)
    realization = from_samples(over(component), batch)
    out = integral(u, realization)
    assert jnp.allclose(jnp.asarray(out.data), 5.0)
    data = jnp.arange(6.0, dtype="float64")
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

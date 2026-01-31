#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.random as jr
import pytest

from phydrax.domain import (
    Boundary,
    FixedStart,
    Interval1d,
    ProductStructure,
    TimeInterval,
)
from phydrax.domain._structure import CoordSeparableBatch


def test_product_domain_sampling_produces_labeled_points_batch():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 2.0)
    dom = geom @ time

    component = dom.component()
    structure = ProductStructure((("x",), ("t",)))
    batch = component.sample((3, 4), structure=structure, key=jr.key(0))

    axis_x = batch.structure.axis_for("x")
    axis_t = batch.structure.axis_for("t")
    assert axis_x is not None and axis_t is not None
    assert axis_x.startswith("__phydra_blk__")
    assert axis_t.startswith("__phydra_blk__")
    assert axis_x != axis_t

    x = batch["x"]
    t = batch["t"]
    assert x.dims == (axis_x, None)
    assert t.dims == (axis_t,)
    assert x.data.shape == (3, 1)
    assert t.data.shape == (4,)


def test_fixed_start_excludes_time_axis_from_structure():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 2.0)
    dom = geom @ time

    component = dom.component({"t": FixedStart()})
    structure = ProductStructure((("x",),))
    batch = component.sample(5, structure=structure, key=jr.key(0))

    axis_x = batch.structure.axis_for("x")
    assert axis_x is not None
    assert batch.structure.axis_for("t") is None
    assert batch["t"].dims == ()


def test_coord_separable_sampling_for_geometry_label():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 2.0)
    dom = geom @ time

    component = dom.component()
    dense_structure = ProductStructure((("t",),))
    batch = component.sample_coord_separable(
        {"x": 4},
        num_points=3,
        dense_structure=dense_structure,
        key=jr.key(0),
    )

    assert isinstance(batch, CoordSeparableBatch)
    assert isinstance(batch["x"], tuple)
    assert len(batch["x"]) == 1
    assert batch.coord_axes_by_label["x"][0].startswith("__phydra_sep__x__")
    assert batch["x"][0].dims == batch.coord_axes_by_label["x"]
    assert batch.coord_mask_by_label["x"].dims == batch.coord_axes_by_label["x"]
    assert batch["t"].dims[0].startswith("__phydra_blk__t")


def test_coord_separable_sampling_rejects_boundary_component():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 2.0)
    dom = geom @ time

    component = dom.component({"x": Boundary()})
    dense_structure = ProductStructure((("t",),))
    with pytest.raises(ValueError):
        component.sample_coord_separable(
            {"x": 4},
            num_points=3,
            dense_structure=dense_structure,
            key=jr.key(0),
        )

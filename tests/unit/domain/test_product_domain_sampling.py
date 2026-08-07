#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.random as jr
import numpy as np
import pytest
from scipy.stats.qmc import Sobol

import phydrax as phx
from phydrax.domain import (
    Boundary,
    ComponentSum,
    DatasetDomain,
    FixedStart,
    GridBatch,
    HyperRectangle,
    Interior,
    Interval1d,
    SampleLayout,
    TimeInterval,
)
from phydrax.sampling import SobolDesign


def test_product_domain_sampling_produces_labeled_points_batch():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 2.0)
    dom = geom @ time

    component = dom.component()
    structure = SampleLayout((("x",), ("t",)))
    batch = component.sample(
        phx.domain.PointSampling((3, 4), layout=structure), key=jr.key(0)
    )

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


def test_same_block_sobol_uses_one_joint_reference_design():
    domain = TimeInterval(0.0, 1.0).relabel("x") @ TimeInterval(0.0, 1.0)
    structure = SampleLayout((("x", "t"),))

    batch = domain.component().sample(
        phx.domain.PointSampling(8, layout=structure, design=SobolDesign()), key=jr.key(0)
    )

    actual = np.column_stack((batch["x"].data, batch["t"].data))
    expected = Sobol(2, scramble=False).random(8)

    assert np.array_equal(actual, expected)
    assert np.mean((actual[:, 0] - actual[:, 1]) ** 2) > 0.0


def test_joint_design_slices_multidimensional_reference_transports():
    box = HyperRectangle([1.0, 10.0], [3.0, 14.0], label="x")
    domain = box @ TimeInterval(-1.0, 1.0)
    structure = SampleLayout((("x", "t"),))

    batch = domain.component().sample(
        phx.domain.PointSampling(8, layout=structure, design=SobolDesign()), key=jr.key(0)
    )

    unit = Sobol(3, scramble=False).random(8)
    expected_x = np.column_stack(
        (
            1.0 + 2.0 * unit[:, 0],
            10.0 + 4.0 * unit[:, 1],
        )
    )
    expected_t = -1.0 + 2.0 * unit[:, 2]

    assert np.array_equal(batch["x"].data, expected_x)
    assert np.array_equal(batch["t"].data, expected_t)


def test_joint_design_preserves_finite_dataset_rows():
    dataset = DatasetDomain(
        {"value": np.asarray([10.0, 20.0, 30.0, 40.0])},
        label="data",
    )
    domain = dataset @ TimeInterval(0.0, 1.0)
    structure = SampleLayout((("data", "t"),))

    batch = domain.component().sample(
        phx.domain.PointSampling(8, layout=structure, design=SobolDesign()), key=jr.key(0)
    )

    unit = Sobol(2, scramble=False).random(8)
    indices = np.floor(4 * unit[:, 0]).astype(int)

    assert np.array_equal(
        batch["data"]["value"].data,
        np.asarray([10.0, 20.0, 30.0, 40.0])[indices],
    )
    assert np.array_equal(batch["t"].data, unit[:, 1])


def test_fixed_start_excludes_time_axis_from_structure():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 2.0)
    dom = geom @ time

    component = dom.component({"t": FixedStart()})
    structure = SampleLayout((("x",),))
    batch = component.sample(phx.domain.PointSampling(5, layout=structure), key=jr.key(0))

    axis_x = batch.structure.axis_for("x")
    assert axis_x is not None
    assert batch.structure.axis_for("t") is None
    assert batch["t"].dims == ()


def test_coord_separable_sampling_for_geometry_label():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 2.0)
    dom = geom @ time

    component = dom.component()
    dense_structure = SampleLayout((("t",),))
    batch = component.sample(
        phx.domain.GridSampling(
            {"x": 4}, dense=phx.domain.PointSampling(3, layout=dense_structure)
        ),
        key=jr.key(0),
    )

    assert isinstance(batch, GridBatch)
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
    dense_structure = SampleLayout((("t",),))
    with pytest.raises(ValueError):
        component.sample(
            phx.domain.GridSampling(
                {"x": 4}, dense=phx.domain.PointSampling(3, layout=dense_structure)
            ),
            key=jr.key(0),
        )


def test_product_boundary_is_additive_component_collection():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    boundary = domain.boundary()

    assert isinstance(boundary, ComponentSum)
    assert len(boundary.terms) == 3
    assert all(term.domain.same_support(domain) for term in boundary.terms)
    assert float(boundary.mass.value) == pytest.approx(4.0)


def test_component_collection_rejects_invalid_terms():
    domain = Interval1d(0.0, 1.0)
    term = domain.component({"x": Interior()})
    incompatible = Interval1d(0.0, 2.0).component({"x": Interior()})

    with pytest.raises(ValueError, match="non-empty"):
        ComponentSum(())
    with pytest.raises(ValueError, match="duplicates"):
        ComponentSum((term, term))
    with pytest.raises(ValueError, match="compatible labeled domain"):
        ComponentSum((term, incompatible))

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.random as jr
import numpy as np
import pytest
from scipy.stats.qmc import Sobol

from phydrax.domain import (
    Boundary,
    DatasetDomain,
    DomainComponentUnion,
    FixedStart,
    HyperRectangle,
    Interior,
    Interval1d,
    ProductStructure,
    TimeInterval,
)
from phydrax.domain._structure import CoordSeparableBatch
from phydrax.sampling import SobolDesign


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


def test_same_block_sobol_uses_one_joint_reference_design():
    domain = TimeInterval(0.0, 1.0).relabel("x") @ TimeInterval(0.0, 1.0)
    structure = ProductStructure((("x", "t"),))

    batch = domain.component().sample(
        8,
        structure=structure,
        sampler=SobolDesign(),
        key=jr.key(0),
    )

    actual = np.column_stack((batch["x"].data, batch["t"].data))
    expected = Sobol(2, scramble=False).random(8)

    assert np.array_equal(actual, expected)
    assert np.mean((actual[:, 0] - actual[:, 1]) ** 2) > 0.0


def test_joint_design_slices_multidimensional_reference_transports():
    box = HyperRectangle([1.0, 10.0], [3.0, 14.0], label="x")
    domain = box @ TimeInterval(-1.0, 1.0)
    structure = ProductStructure((("x", "t"),))

    batch = domain.component().sample(
        8,
        structure=structure,
        sampler=SobolDesign(),
        key=jr.key(0),
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
    structure = ProductStructure((("data", "t"),))

    batch = domain.component().sample(
        8,
        structure=structure,
        sampler=SobolDesign(),
        key=jr.key(0),
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


def test_product_boundary_is_additive_component_collection():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    boundary = domain.boundary()

    assert isinstance(boundary, DomainComponentUnion)
    assert len(boundary.terms) == 3
    assert all(term.domain.equivalent(domain) for term in boundary.terms)
    assert float(boundary.measure()) == pytest.approx(4.0)


def test_component_collection_rejects_invalid_terms():
    domain = Interval1d(0.0, 1.0)
    term = domain.component({"x": Interior()})
    incompatible = Interval1d(0.0, 2.0).component({"x": Interior()})

    with pytest.raises(ValueError, match="non-empty"):
        DomainComponentUnion(())
    with pytest.raises(ValueError, match="duplicates"):
        DomainComponentUnion((term, term))
    with pytest.raises(ValueError, match="compatible labeled domain"):
        DomainComponentUnion((term, incompatible))

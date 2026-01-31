#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.random as jr
import pytest

from phydrax.constraints import ContinuousInitialConstraint, FunctionalConstraint
from phydrax.domain import (
    DomainComponentUnion,
    Interior,
    Interval1d,
    ProductStructure,
    TimeInterval,
)


def test_coord_separable_rejects_component_union():
    geom = Interval1d(0.0, 1.0)
    c1 = geom.component({"x": Interior()})
    c2 = geom.component({"x": Interior()})
    union = DomainComponentUnion((c1, c2))

    with pytest.raises(ValueError, match="coord_separable sampling is not supported"):
        FunctionalConstraint.from_operator(
            component=union,
            operator=lambda u: u,
            constraint_vars="u",
            num_points=4,
            structure=ProductStructure((("x",),)),
            coord_separable={"x": 4},
        )


def test_coord_separable_requires_dense_structure_for_other_labels():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component()

    constraint = FunctionalConstraint.from_operator(
        component=component,
        operator=lambda u: u,
        constraint_vars="u",
        num_points=4,
        structure=ProductStructure((("x",),)),
        coord_separable={"x": 4},
    )

    with pytest.raises(ValueError, match="dense_structure"):
        constraint.sample(key=jr.key(0))


def test_continuous_initial_requires_fixed_start():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"t": Interior()})

    with pytest.raises(ValueError, match="FixedStart"):
        ContinuousInitialConstraint(
            "u",
            component,
            func=0.0,
            num_points=4,
            structure=ProductStructure((("x",),)),
        )


def test_domain_join_rejects_label_collision():
    left = Interval1d(0.0, 1.0)
    right = Interval1d(-1.0, 2.0)
    with pytest.raises(ValueError, match="Label collision"):
        _ = left @ right

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax.random as jr
import pytest

import phydrax as phx
from phydrax.domain import SampleLayout


@pytest.fixture
def box3d():
    return phx.domain.GeometryDomain(
        phx.geometry.Box(
            center=(0.0, 0.0, 0.0),
            size=(2.0, 2.0, 2.0),
            feature_id="operator-test-box",
        ).compile()
    )


@pytest.fixture
def sample_batch():
    def _sample(component, /, *, blocks, num_points, key=0, sampler="latin_hypercube"):
        structure = SampleLayout(blocks=blocks)
        return component.sample(
            phx.domain.PointSampling(num_points, layout=structure, design=sampler),
            key=jr.key(int(key)),
        )

    return _sample


@pytest.fixture
def sample_grid():
    def _sample(
        component,
        coord_separable,
        /,
        *,
        num_points=(),
        dense_blocks=(),
        key=0,
        sampler="latin_hypercube",
    ):
        dense_structure = (
            SampleLayout(blocks=dense_blocks) if dense_blocks is not None else None
        )
        return component.sample(
            phx.domain.GridSampling(
                coord_separable,
                dense=phx.domain.PointSampling(
                    num_points, layout=dense_structure, design=sampler
                ),
                design=sampler,
            ),
            key=jr.key(int(key)),
        )

    return _sample

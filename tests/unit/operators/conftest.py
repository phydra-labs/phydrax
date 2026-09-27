#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.random as jr
import pytest

import phydrax as phx
from phydrax.domain import SampleLayout


@pytest.fixture
def box3d() -> Any:
    return phx.domain.GeometryDomain(
        phx.geometry.Box(
            center=(0.0, 0.0, 0.0),
            size=(2.0, 2.0, 2.0),
            feature_id="operator-test-box",
        ).compile()
    )


@pytest.fixture
def sample_batch() -> Any:
    def _sample(
        component: Any,
        /,
        *,
        blocks: Any,
        num_points: Any,
        key: Any = 0,
        sampler: Any = "latin_hypercube",
    ) -> Any:
        structure = SampleLayout(blocks=blocks)
        return component.sample(
            phx.domain.PointSampling(num_points, layout=structure, design=sampler),
            key=jr.key(int(key)),
        )

    return _sample


@pytest.fixture
def sample_grid() -> Any:
    def _sample(
        component: Any,
        coord_separable: Any,
        /,
        *,
        num_points: Any = (),
        dense_blocks: Any = (),
        key: Any = 0,
        sampler: Any = "latin_hypercube",
    ) -> Any:
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

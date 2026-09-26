#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.discretization.vortex._diffusion_complete import (
    GaussianPSENeighborhoodPlan,
)


def _pairs() -> phx.discretization.particle.ParticlePairRelation:
    relation = phx.sparse.EdgeRelation(
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        source_size=2,
        target_size=2,
    )
    return phx.discretization.particle.ParticlePairRelation(
        relation,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        source_support_id="particles",
        target_support_id="particles",
        same_set=True,
        unordered=True,
    )


@pytest.mark.parametrize(
    ("periodic_axes", "domain"),
    [((True, False), "periodic"), ((False, False), "free-space")],
)
def test_pse_neighborhood_domain_follows_box_periodicity(
    periodic_axes: tuple[bool, bool], domain: str
) -> None:
    box = phx.discretization.particle.ParticleBox(
        jnp.zeros((2,)), jnp.ones((2,)), periodic_axes=periodic_axes
    )
    plan = GaussianPSENeighborhoodPlan(_pairs(), 2, 0.1, box=box)
    assert plan.capabilities.domain == domain

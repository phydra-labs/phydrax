# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._strict import StrictModule
from phydrax.discretization.contact._participant import FunctionContactParticipant
from phydrax.discretization.contact._surface import CollisionSurfacePlan
from phydrax.linalg import ArraySpace


@final
class _ConditionedPositions(StrictModule):
    matrix: Array

    def __call__(self, coordinates: Array, /) -> Array:
        return (self.matrix @ coordinates)[None, :]


@pytest.mark.strict_jax
def test_conditioned_geometry_callbacks_remain_dynamic_under_compilation() -> None:
    space = ArraySpace((2,), dtype=jnp.float64)
    surface = CollisionSurfacePlan(
        jnp.asarray((41,), dtype=jnp.int64),
        ambient_dimension=2,
        allow_isolated_vertices=True,
    )
    action = _ConditionedPositions(
        jnp.asarray(((2.0, 0.0), (0.0, 3.0)), dtype=jnp.float64)
    )
    participant = FunctionContactParticipant(
        surface, space, action, tangent_space=space, participant_id="conditioned-geometry"
    )
    q = jnp.asarray((0.25, 0.5), dtype=jnp.float64)
    changed = eqx.tree_at(lambda owner: owner.matrix, action, 2.0 * action.matrix)
    refreshed = eqx.tree_at(lambda owner: owner.position_action, participant, changed)
    positions = jax.jit(lambda owner, coordinates: owner.positions(coordinates))
    np.testing.assert_allclose(positions(participant, q), ((0.5, 1.5),), atol=1e-14)
    np.testing.assert_allclose(positions(refreshed, q), ((1.0, 3.0),), atol=1e-14)
    assert participant.participant_id == refreshed.participant_id
    derivative = eqx.filter_grad(lambda owner: jnp.sum(owner.positions(q)))(participant)
    derivative_action = derivative.position_action
    if not isinstance(derivative_action, _ConditionedPositions):
        raise TypeError("Conditioned position derivative lost its callback contract.")
    np.testing.assert_allclose(
        derivative_action.matrix, ((0.25, 0.5), (0.25, 0.5)), atol=1e-14
    )
    direction = jnp.asarray((0.2, -0.1), dtype=jnp.float64)
    effort = jnp.asarray(((1.2, -0.4),), dtype=jnp.float64)
    assert bool(refreshed.duality_evidence(q, direction, effort).valid)

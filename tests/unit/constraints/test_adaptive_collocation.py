#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr

from phydrax.constraints import (
    FunctionalConstraint,
    PeriodicCollocation,
    R3,
    RARD,
)
from phydrax.domain import Interval1d, PointsBatch, ProductStructure


def _interval_constraint(policy, *, num_points=32):
    domain = Interval1d(0.0, 1.0)
    structure = ProductStructure((("x",),))

    @domain.Function("x")
    def coordinate(x):
        return x[0]

    constraint = FunctionalConstraint.from_operator(
        component=domain.component(),
        operator=lambda _u: coordinate,
        constraint_vars="u",
        num_points=num_points,
        structure=structure,
        sampler="uniform",
        collocation_policy=policy,
    )
    return domain, constraint, {"u": domain.Function()(0.0)}




def _coordinates(population):
    field = population.batch.points["x"]
    assert isinstance(field, cx.Field)
    return jnp.asarray(field.data).reshape((-1,))


def test_periodic_collocation_replaces_a_fixed_size_population():
    policy = PeriodicCollocation(refresh_every=2, sampler="uniform")
    _domain, constraint, functions = _interval_constraint(policy)
    initial = policy.initialize(constraint, key=jr.key(0))
    refreshed = policy.refresh(
        constraint, functions, initial, key=jr.key(1), iter_=2
    )
    assert isinstance(initial.batch, PointsBatch)
    assert refreshed.batch.structure == initial.batch.structure
    assert _coordinates(refreshed).shape == _coordinates(initial).shape
    assert not jnp.allclose(_coordinates(refreshed), _coordinates(initial))




def test_r3_retains_difficult_points_and_preserves_population_size():
    policy = R3(refresh_every=1, sampler="uniform")
    _domain, constraint, functions = _interval_constraint(policy, num_points=64)
    initial = policy.initialize(constraint, key=jr.key(4))
    initial_x = _coordinates(initial)
    refreshed = policy.refresh(
        constraint, functions, initial, key=jr.key(5), iter_=1
    )
    refreshed_x = _coordinates(refreshed)
    assert refreshed_x.shape == initial_x.shape
    difficult = initial_x[initial_x * initial_x > jnp.mean(initial_x * initial_x)]
    assert jnp.all(jnp.isin(difficult, refreshed_x))










def test_fixed_capacity_rar_d_activates_new_slots():
    policy = RARD(
        refresh_every=1,
        sampler="uniform",
        initial_active_fraction=0.5,
        refinement_fraction=0.25,
    )
    _domain, constraint, functions = _interval_constraint(policy, num_points=40)
    initial = policy.initialize(constraint, key=jr.key(12))
    assert initial.active is not None
    before = int(jnp.sum(jnp.asarray(initial.active.data)))
    refreshed = policy.refresh(
        constraint, functions, initial, key=jr.key(13), iter_=1
    )
    assert refreshed.active is not None
    assert int(jnp.sum(jnp.asarray(refreshed.active.data))) == before + 10
    assert _coordinates(refreshed).shape == (40,)

















#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from phydrax.constraints import (
    ContinuousDirichletBoundaryConstraint,
    ContinuousInitialConstraint,
    FunctionalConstraint,
    IntegralEqualityConstraint,
)
from phydrax.constraints._continuous_interior import ContinuousPointwiseInteriorConstraint
from phydrax.domain import (
    Boundary,
    FixedStart,
    FourierAxisSpec,
    Interval1d,
    PointsBatch,
    ProductStructure,
    TimeInterval,
)
from phydrax.domain._function import BatchAwareCallable
from phydrax.domain._structure import CoordSeparableBatch


class _KeyConsumingResidual(BatchAwareCallable):
    def __call_batch__(self, batch, /, *, key, **kwargs):
        del kwargs
        reference = batch["x"]
        draw = jr.uniform(key)
        return cx.Field(
            jnp.broadcast_to(draw, reference.data.shape),
            dims=reference.dims,
        )


def _sum_fields(tree) -> jnp.ndarray:
    leaves = jtu.tree_leaves(tree, is_leaf=lambda x: isinstance(x, cx.Field))
    total = jnp.array(0.0, dtype=float)
    for leaf in leaves:
        if isinstance(leaf, cx.Field):
            total = total + jnp.sum(leaf.data)
    return total


def _jit_sample_sum(constraint):
    def _sample_sum(key):
        batch = constraint.sample(key=key)
        if isinstance(batch, tuple):
            total = jnp.array(0.0, dtype=float)
            for item in batch:
                total = total + _sum_fields(item.points)
            return total
        if isinstance(batch, (PointsBatch, CoordSeparableBatch)):
            return _sum_fields(batch.points)
        return _sum_fields(batch)

    return eqx.filter_jit(_sample_sum)(jr.key(0))


def test_sampling_jit_boundary_constraint():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    structure = ProductStructure((("x",),))

    constraint = ContinuousDirichletBoundaryConstraint(
        "u",
        component,
        target=0.0,
        num_points=8,
        structure=structure,
    )
    total = _jit_sample_sum(constraint)
    assert jnp.isfinite(total)


def test_sampling_jit_initial_constraint():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"t": FixedStart()})
    structure = ProductStructure((("x",),))

    constraint = ContinuousInitialConstraint(
        "u",
        component,
        func=0.0,
        num_points=8,
        structure=structure,
    )
    total = _jit_sample_sum(constraint)
    assert jnp.isfinite(total)


def test_sampling_jit_interior_constraint():
    geom = Interval1d(0.0, 1.0)
    structure = ProductStructure((("x",),))

    constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda u: u,
        num_points=8,
        structure=structure,
    )
    total = _jit_sample_sum(constraint)
    assert jnp.isfinite(total)


def test_sampling_jit_interior_constraint_coord_separable_fourier_axis_spec():
    geom = Interval1d(0.0, 1.0)
    structure = ProductStructure((("x",),))

    constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda u: u,
        num_points={"x": FourierAxisSpec(8)},
        structure=structure,
    )
    total = _jit_sample_sum(constraint)
    assert jnp.isfinite(total)


def test_sampling_jit_integral_constraint():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    structure = ProductStructure((("x",),))

    constraint = IntegralEqualityConstraint.from_operator(
        component=component,
        operator=lambda u: u,
        constraint_vars="u",
        num_points=8,
        structure=structure,
    )
    total = _jit_sample_sum(constraint)
    assert jnp.isfinite(total)


def test_functional_constraint_splits_sampling_and_evaluation_keys():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    structure = ProductStructure((("x",),))
    function = geom.Function("x")(_KeyConsumingResidual())
    constraint = FunctionalConstraint.from_operator(
        component=component,
        operator=lambda u: u,
        constraint_vars="u",
        num_points=8,
        structure=structure,
    )
    caller_key = jr.key(21)
    _, evaluation_key = jr.split(caller_key)

    sampled_loss = constraint.loss({"u": function}, key=caller_key)
    supplied_batch = constraint.sample(key=jr.key(22))
    supplied_loss = constraint.loss(
        {"u": function},
        key=caller_key,
        batch=supplied_batch,
    )

    assert jnp.allclose(sampled_loss, jr.uniform(evaluation_key) ** 2, atol=1e-14)
    assert jnp.allclose(supplied_loss, jr.uniform(caller_key) ** 2, atol=1e-14)


def test_integral_constraint_splits_sampling_and_evaluation_keys():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    structure = ProductStructure((("x",),))
    function = geom.Function("x")(_KeyConsumingResidual())
    constraint = IntegralEqualityConstraint.from_operator(
        component=component,
        operator=lambda u: u,
        constraint_vars="u",
        num_points=8,
        structure=structure,
    )
    caller_key = jr.key(23)
    _, evaluation_key = jr.split(caller_key)

    loss = constraint.loss({"u": function}, key=caller_key)

    assert jnp.allclose(loss, jr.uniform(evaluation_key) ** 2, atol=1e-14)

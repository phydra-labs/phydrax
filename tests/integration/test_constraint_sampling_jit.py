#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

import phydrax as phx
from phydrax.constraints import (
    ContinuousDirichletBoundaryConstraint,
    ContinuousInitialConstraint,
    FunctionalConstraint,
    IntegralEqualityConstraint,
)
from phydrax.constraints._continuous_interior import ContinuousPointwiseInteriorConstraint
from phydrax.domain import (
    BatchEvaluator,
    Boundary,
    FixedStart,
    FourierAxisSpec,
    GridBatch,
    Interval1d,
    PointBatch,
    SampleLayout,
    TimeInterval,
)


class _KeyConsumingResidual(BatchEvaluator):
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
        if isinstance(batch, (PointBatch, GridBatch)):
            return _sum_fields(batch.points)
        return _sum_fields(batch)

    return eqx.filter_jit(_sample_sum)(jr.key(0))


def test_sampling_jit_boundary_constraint():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    structure = SampleLayout((("x",),))

    constraint = ContinuousDirichletBoundaryConstraint(
        "u",
        component,
        target=0.0,
        sampling=phx.domain.PointSampling(8, layout=structure),
    )
    total = _jit_sample_sum(constraint)
    assert jnp.isfinite(total)


def test_sampling_jit_initial_constraint():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"t": FixedStart()})
    structure = SampleLayout((("x",),))

    constraint = ContinuousInitialConstraint(
        "u",
        component,
        func=0.0,
        sampling=phx.domain.PointSampling(8, layout=structure),
    )
    total = _jit_sample_sum(constraint)
    assert jnp.isfinite(total)


def test_sampling_jit_interior_constraint():
    geom = Interval1d(0.0, 1.0)
    structure = SampleLayout((("x",),))

    constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda u: u,
        sampling=phx.domain.PointSampling(8, layout=structure),
    )
    total = _jit_sample_sum(constraint)
    assert jnp.isfinite(total)


def test_sampling_jit_interior_constraint_coord_separable_fourier_axis_spec():
    geom = Interval1d(0.0, 1.0)
    structure = SampleLayout((("x",),))

    constraint = ContinuousPointwiseInteriorConstraint("u",
    geom,
    operator=lambda u: u, sampling=phx.domain.GridSampling({"x": FourierAxisSpec(8)}), )
    total = _jit_sample_sum(constraint)
    assert jnp.isfinite(total)


def test_sampling_jit_integral_constraint():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    structure = SampleLayout((("x",),))

    constraint = IntegralEqualityConstraint.from_operator(component=component,
    operator=lambda u: u,
    constraint_vars="u", sampling=phx.domain.PointSampling(8, layout=structure), )
    total = _jit_sample_sum(constraint)
    assert jnp.isfinite(total)


def test_functional_constraint_splits_sampling_and_evaluation_keys():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    structure = SampleLayout((("x",),))
    function = geom.Function("x")(_KeyConsumingResidual())
    constraint = FunctionalConstraint.from_operator(component=component,
    operator=lambda u: u,
    constraint_vars="u", sampling=phx.domain.PointSampling(8, layout=structure), )
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
    structure = SampleLayout((("x",),))
    function = geom.Function("x")(_KeyConsumingResidual())
    constraint = IntegralEqualityConstraint.from_operator(component=component,
    operator=lambda u: u,
    constraint_vars="u", sampling=phx.domain.PointSampling(8, layout=structure), )
    caller_key = jr.key(23)
    _, evaluation_key = jr.split(caller_key)

    loss = constraint.loss({"u": function}, key=caller_key)

    assert jnp.allclose(loss, jr.uniform(evaluation_key) ** 2, atol=1e-14)


def test_geometry_domain_point_and_grid_sampling_share_constraint_contract():
    geometry = phx.domain.GeometryDomain(
        phx.geometry.Square(
            center=(0.0, 0.0),
            side=2.0,
            feature_id="domain-substrate-integration",
        ).compile()
    )
    domain = geometry @ phx.domain.TimeInterval(0.0, 1.0)
    component = domain.component()
    point_plan = phx.domain.PointSampling(
        16,
        layout=phx.domain.SampleLayout((("x", "t"),)),
    )

    @domain.Function("x", "t")
    def exact(x, t):
        return jnp.sum(x**2) + t

    residual = lambda field: (
        phx.operators.laplacian(field, var="x")
        + phx.operators.partial_t(field, var="t")
        - 5.0
    )
    constraint = FunctionalConstraint.from_operator(
        component=component,
        operator=residual,
        constraint_vars="u",
        sampling=point_plan,
    )

    point_batch = component.sample(point_plan, key=jr.key(31))
    assert isinstance(point_batch, PointBatch)
    assert exact(point_batch).data.shape == (16,)
    assert constraint.loss({"u": exact}, key=jr.key(32)) < 1e-12

    grid_batch = component.sample(
        phx.domain.GridSampling(
            {"x": (6, 5)},
            dense=phx.domain.PointSampling(
                3,
                layout=phx.domain.SampleLayout((("t",),)),
            ),
        ),
        key=jr.key(33),
    )
    assert isinstance(grid_batch, GridBatch)
    assert exact(grid_batch).data.shape == (3, 6, 5)

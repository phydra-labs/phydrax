#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic seams and initial data stay exact through accepted solver updates.

The seam relation is checked independently of the periodic residual evaluator:
values and JAX coordinate derivatives of the enforced field are compared at the
two identified faces. Exactness is a structural claim, so the checks do not
depend on optimization progress.
"""

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax import Array

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.conditions import Periodic
from phydrax.domain import (
    DomainFunction,
    FixedStart,
    Interval1d,
    PeriodicIdentification,
    PointBatch,
    PointSampling,
    SampleLayout,
    TimeInterval,
)
from phydrax.enforcement import (
    EnforcementOptions,
    EnforcementSpec,
    prepare_periodic_projection,
    PreparedPeriodicProjection,
)
from phydrax.nn.models import MLP
from phydrax.operators.differential import dt, laplacian
from phydrax.solver import FunctionalSolver


DOMAIN = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 0.1)
TIMES = jnp.asarray([0.0, 0.03, 0.07, 0.1])


def _paired_batch(xs: Array, ts: Array) -> PointBatch:
    structure = SampleLayout((("x", "t"),)).canonicalize(DOMAIN.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.AxisArray(jnp.reshape(xs, (-1, 1)), dims=(axis, None)),
            "t": cx.AxisArray(jnp.reshape(ts, (-1,)), dims=(axis,)),
        }
    )
    return PointBatch(points=points, structure=structure)


def _value(field: DomainFunction, x: Array, t: Array) -> Array:
    batch = _paired_batch(jnp.reshape(x, (1,)), jnp.reshape(t, (1,)))
    return jnp.asarray(field(batch).data).reshape(())


def _assert_seam_and_initial_exact(fields: Mapping[str, Any]) -> None:
    field = fields["u"]
    slope = jax.grad(_value, argnums=1)
    for t in TIMES:
        lower, upper = jnp.asarray(0.0), jnp.asarray(1.0)
        assert abs(float(_value(field, upper, t) - _value(field, lower, t))) < 1e-12
        assert abs(float(slope(field, upper, t) - slope(field, lower, t))) < 1e-11
    xs = jnp.asarray([0.0, 0.21, 0.5, 0.83, 1.0])
    initial = jnp.asarray(field(_paired_batch(xs, jnp.zeros_like(xs))).data)
    np.testing.assert_allclose(
        initial.reshape((-1,)), jnp.sin(2.0 * jnp.pi * xs), rtol=0.0, atol=1e-12
    )


def test_functional_solver_keeps_periodic_seams_and_initial_data_exact() -> None:
    seam = PeriodicIdentification(DOMAIN, "x")
    model = MLP(
        in_size=2,
        out_size="scalar",
        width_size=12,
        depth=2,
        activation=jnp.tanh,
        key=jr.key(0),
    )
    functions = {"u": DOMAIN.Model("x", "t")(model)}
    declarations = (
        Periodic("u", seam.pairing(), label="value"),
        Periodic("u", seam.pairing(), order=1, label="slope"),
    )
    prepared = prepare_periodic_projection(
        functions, declarations, route="analytic", data_compatibility="probed"
    )
    initial = phx.conditions.Initial(
        "u",
        DOMAIN.component({"t": FixedStart()}),
        target=DOMAIN.Function("x")(lambda x: jnp.sin(2.0 * jnp.pi * x[0])),
    )
    program = phx.enforcement.compile(
        functions,
        (
            EnforcementSpec(initial),
            EnforcementSpec(prepared.condition, realization=prepared),
        ),
        options=EnforcementOptions(num_reference=256),
    )
    (bound,) = program.realization_specs
    assert isinstance(bound.realization, PreparedPeriodicProjection)
    assert [
        (record.contract_id, record.status)
        for record in bound.realization.evidence.preserved
    ] == [("Initial:u:order-0", "probed")]

    heat = phx.conditions.Residual(
        "u",
        DOMAIN.component(),
        lambda u: dt(u, var="t") - 0.1 * laplacian(u, var="x"),
    )
    term = phx.terms.ResidualPenalty(
        heat,
        phx.integration.per_step(phx.integration.mean_over(heat.on), PointSampling(32)),
    )
    solver = FunctionalSolver(functions=functions, terms=(term,), enforcement=program)
    _assert_seam_and_initial_exact(solver.ansatz_functions())

    solved = solver.solve(num_iter=3, optim=optax.adam(1e-2), seed=0, keep_best=False)

    before = jax.tree.leaves(eqx.filter(solver.functions, eqx.is_inexact_array))
    after = jax.tree.leaves(eqx.filter(solved.functions, eqx.is_inexact_array))
    assert any(
        not bool(jnp.array_equal(old, new))
        for old, new in zip(before, after, strict=True)
    )
    assert bool(jnp.isfinite(solved.loss(key=jr.key(1))))
    _assert_seam_and_initial_exact(solved.ansatz_functions())


def test_functional_solver_keeps_coordinate_face_walls_and_seam_exact() -> None:
    box = phx.domain.HyperRectangle(
        jnp.zeros((2,), dtype=jnp.float64), jnp.ones((2,), dtype=jnp.float64)
    )
    seam = PeriodicIdentification(box, "x", component=0)
    model = MLP(
        in_size=2,
        out_size="scalar",
        width_size=12,
        depth=2,
        activation=jnp.tanh,
        key=jr.key(0),
    )
    functions = {"u": box.Model("x")(model)}
    prepared = prepare_periodic_projection(
        functions, (Periodic("u", seam.pairing()),), route="analytic"
    )
    walls = phx.domain.physical_boundary(box, (seam,))
    program = phx.enforcement.compile(
        functions,
        (
            *(
                EnforcementSpec(phx.conditions.Dirichlet("u", wall, target=0.0))
                for wall in walls
            ),
            EnforcementSpec(prepared.condition, realization=prepared),
        ),
        options=EnforcementOptions(num_reference=256),
    )
    poisson = phx.conditions.Residual(
        "u", box.component(), lambda u: laplacian(u, var="x") + 1.0
    )
    term = phx.terms.ResidualPenalty(
        poisson,
        phx.integration.per_step(
            phx.integration.mean_over(poisson.on), PointSampling(32)
        ),
    )
    solver = FunctionalSolver(functions=functions, terms=(term,), enforcement=program)

    solved = solver.solve(num_iter=2, optim=optax.adam(1e-2), seed=0, keep_best=False)

    field = solved.ansatz_functions()["u"]
    s = jnp.asarray([0.0, 0.17, 0.5, 0.91, 1.0])

    def values(points: Array) -> Array:
        batch = box.component().points({"x": points})
        return jnp.asarray(field(batch).data).reshape((-1,))

    for wall_y in (0.0, 1.0):
        on_wall = values(jnp.stack((s, jnp.full_like(s, wall_y)), axis=-1))
        np.testing.assert_allclose(on_wall, 0.0, rtol=0.0, atol=1e-12)
    lower = values(jnp.stack((jnp.zeros_like(s), s), axis=-1))
    upper = values(jnp.stack((jnp.ones_like(s), s), axis=-1))
    np.testing.assert_allclose(upper, lower, rtol=0.0, atol=1e-12)

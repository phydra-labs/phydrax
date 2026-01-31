#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax._frozendict import frozendict
from phydrax.constraints import (
    enforce_dirichlet,
    enforce_neumann,
    enforce_robin,
    enforce_sommerfeld,
    enforce_traction,
)
from phydrax.domain import (
    Boundary,
    FixedStart,
    Interval1d,
    PointsBatch,
    ProductStructure,
    Square,
    TimeInterval,
)
from phydrax.operators.differential import directional_derivative, dt, partial_x
from phydrax.operators.integral import mean
from phydrax.solver import (
    EnforcedConstraintPipelines,
    EnforcedInteriorData,
    SingleFieldEnforcedConstraint,
)


def _line_batch(domain, xs):
    structure = ProductStructure((("x",),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {"x": cx.Field(jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None))}
    )
    return PointsBatch(points=points, structure=structure)


def _paired_batch(domain, xs, ts):
    structure = ProductStructure((("x", "t"),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.Field(
                jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None)
            ),
            "t": cx.Field(jnp.asarray(ts, dtype=float).reshape((-1,)), dims=(axis,)),
        }
    )
    return PointsBatch(points=points, structure=structure)


def test_error_missing_anchor_label():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + t

    interior = EnforcedInteriorData(
        "u",
        points={"x": jnp.array([[0.25]], dtype=float)},
        values=jnp.array([1.0], dtype=float),
    )

    with pytest.raises(KeyError, match="missing labels"):
        EnforcedConstraintPipelines.build(functions={"u": u}, interior_data=[interior])


def test_error_anchor_on_boundary():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + t

    boundary = domain.component({"x": Boundary()})
    boundary_constraint = SingleFieldEnforcedConstraint(
        "u",
        boundary,
        lambda f: enforce_dirichlet(f, boundary, var="x", target=1.0),
    )

    interior = EnforcedInteriorData(
        "u",
        points={
            "x": jnp.array([[0.0]], dtype=float),
            "t": jnp.array([0.5], dtype=float),
        },
        values=jnp.array([2.0], dtype=float),
    )

    with pytest.raises(ValueError, match="M\\(z_i\\)=0"):
        EnforcedConstraintPipelines.build(
            functions={"u": u},
            constraints=[boundary_constraint],
            interior_data=[interior],
        )


def test_error_anchor_on_initial_time():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + t

    initial = domain.component({"t": FixedStart()})
    initial_constraint = SingleFieldEnforcedConstraint(
        "u",
        initial,
        lambda f: enforce_dirichlet(f, initial, var="t", target=2.0),
        time_derivative_order=0,
    )

    interior = EnforcedInteriorData(
        "u",
        points={
            "x": jnp.array([[0.25]], dtype=float),
            "t": jnp.array([0.0], dtype=float),
        },
        values=jnp.array([2.0], dtype=float),
    )

    with pytest.raises(ValueError, match="M\\(z_i\\)=0"):
        EnforcedConstraintPipelines.build(
            functions={"u": u},
            constraints=[initial_constraint],
            interior_data=[interior],
        )


def test_error_conflicting_duplicate_anchors():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + t

    points = {
        "x": jnp.array([[0.25]], dtype=float),
        "t": jnp.array([0.5], dtype=float),
    }
    a = EnforcedInteriorData("u", points=points, values=jnp.array([1.0], dtype=float))
    b = EnforcedInteriorData("u", points=points, values=jnp.array([2.0], dtype=float))

    with pytest.raises(ValueError, match="Conflicting coincident interior anchors"):
        EnforcedConstraintPipelines.build(functions={"u": u}, interior_data=[a, b])


def test_identity_remainder_toggle_changes_output():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] * 0.0 + 10.0

    left = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    left_constraint = SingleFieldEnforcedConstraint(
        "u",
        left,
        lambda f: enforce_dirichlet(f, left, var="x", target=1.0),
    )

    pipes_no = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=[left_constraint],
        include_identity_remainder=False,
        num_reference=256,
    )
    pipes_yes = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=[left_constraint],
        include_identity_remainder=True,
        num_reference=256,
    )

    u_no = pipes_no.apply({"u": u})["u"]
    u_yes = pipes_yes.apply({"u": u})["u"]
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)

    batch = _line_batch(geom, xs=jnp.array([1.0], dtype=float))
    out_no = eval_jit(u_no, batch).reshape((-1,))[0]
    out_yes = eval_jit(u_yes, batch).reshape((-1,))[0]

    assert jnp.allclose(out_no, 1.0, atol=1e-3)
    assert out_yes > 1.2


def test_enforce_traction_enforces_zero_boundary():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})

    @geom.Function("x")
    def u(x):
        return jnp.array([0.0, 0.0])

    constraint = SingleFieldEnforcedConstraint(
        "u",
        component,
        lambda f: enforce_traction(
            f,
            component,
            var="x",
            lambda_=1.0,
            mu=1.0,
            target=jnp.array([0.0, 0.0]),
        ),
    )

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=[constraint],
        num_reference=128,
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    structure = ProductStructure((("x",),)).canonicalize(geom.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.Field(
                jnp.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float), dims=(axis, None)
            )
        }
    )
    pts = PointsBatch(points=points, structure=structure)
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)
    out = eval_jit(u_enforced, pts)
    assert jnp.allclose(out, 0.0, atol=1e-6)


def test_enforce_neumann_enforces_zero_normal_derivative():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    constraint = SingleFieldEnforcedConstraint(
        "u",
        component,
        lambda f: enforce_neumann(f, component, var="x", target=0.0, mode="forward"),
    )

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=[constraint],
        num_reference=128,
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    n = component.normal(var="x")
    du_dn = directional_derivative(u_enforced, n, var="x", mode="forward")

    structure = ProductStructure((("x",),)).canonicalize(geom.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.Field(
                jnp.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float), dims=(axis, None)
            )
        }
    )
    pts = PointsBatch(points=points, structure=structure)
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)
    out = eval_jit(du_dn, pts)
    assert jnp.allclose(out, 0.0, atol=1e-3)


def test_enforce_robin_enforces_boundary_relation():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})

    @geom.Function("x")
    def u(x):
        return x[0]

    constraint = SingleFieldEnforcedConstraint(
        "u",
        component,
        lambda f: enforce_robin(
            f,
            component,
            var="x",
            dirichlet_coeff=2.0,
            neumann_coeff=1.0,
            target=0.0,
            mode="forward",
        ),
    )

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=[constraint],
        num_reference=128,
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    n = component.normal(var="x")
    du_dn = directional_derivative(u_enforced, n, var="x", mode="forward")
    residual = 2.0 * u_enforced + du_dn

    structure = ProductStructure((("x",),)).canonicalize(geom.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.Field(
                jnp.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float), dims=(axis, None)
            )
        }
    )
    pts = PointsBatch(points=points, structure=structure)
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)
    out = eval_jit(residual, pts)
    assert jnp.allclose(out, 0.0, atol=1e-3)


def test_enforce_sommerfeld_enforces_absorbing_condition():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"x": Boundary()})

    @domain.Function("x", "t")
    def u(x, t):
        return t

    constraint = SingleFieldEnforcedConstraint(
        "u",
        component,
        lambda f: enforce_sommerfeld(
            f,
            component,
            var="x",
            time_var="t",
            wavespeed=1.0,
            target=0.0,
        ),
    )

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        constraints=[constraint],
        num_reference=128,
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    n = component.normal(var="x")
    du_dn = directional_derivative(u_enforced, n, var="x")
    du_dt = dt(u_enforced, var="t")
    residual = du_dn + du_dt

    pts = _paired_batch(
        domain,
        xs=jnp.array([0.0, 1.0], dtype=float),
        ts=jnp.array([0.25, 0.75], dtype=float),
    )
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)
    out = eval_jit(residual, pts)
    assert jnp.allclose(out, 0.0, atol=1e-3)


def test_envelope_extremes():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] * 0.0 + t * 0.0

    anchors = {"x": jnp.array([[0.0]], dtype=float), "t": jnp.array([0.0], dtype=float)}
    values = jnp.array([5.0], dtype=float)

    small = EnforcedInteriorData(
        "u",
        points=anchors,
        values=values,
        use_envelope=True,
        envelope_scale=0.1,
    )
    large = EnforcedInteriorData(
        "u",
        points=anchors,
        values=values,
        use_envelope=True,
        envelope_scale=10.0,
    )

    pipe_small = EnforcedConstraintPipelines.build(
        functions={"u": u}, interior_data=[small]
    )
    pipe_large = EnforcedConstraintPipelines.build(
        functions={"u": u}, interior_data=[large]
    )

    u_small = pipe_small.apply({"u": u})["u"]
    u_large = pipe_large.apply({"u": u})["u"]
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)

    batch = _paired_batch(
        domain, xs=jnp.array([[1.0]], dtype=float), ts=jnp.array([1.0], dtype=float)
    )
    out_small = eval_jit(u_small, batch).reshape((-1,))[0]
    out_large = eval_jit(u_large, batch).reshape((-1,))[0]

    assert out_small < 0.1
    assert out_large > 4.0


def test_equivalent_domain_join_arithmetic():
    dom_a = Interval1d(0.0, 1.0)
    dom_b = Interval1d(0.0, 1.0)

    @dom_a.Function("x")
    def f(x):
        return x[0]

    @dom_b.Function("x")
    def g(x):
        return x[0] * 2.0

    h = f + g
    batch = _line_batch(dom_a, xs=jnp.array([0.2, 0.8], dtype=float))
    eval_jit = eqx.filter_jit(lambda b: h(b).data)
    out = eval_jit(batch).reshape((-1,))
    assert jnp.allclose(out, jnp.array([0.6, 2.4]), atol=1e-6)


def test_join_broadcast_x_t():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @geom.Function("x")
    def f(x):
        return x[0]

    @domain.Function("x", "t")
    def g(x, t):
        return t

    h = f + g
    batch = _paired_batch(
        domain,
        xs=jnp.array([[0.2], [0.8]], dtype=float),
        ts=jnp.array([0.1, 0.9], dtype=float),
    )
    eval_jit = eqx.filter_jit(lambda b: h(b).data)
    out = eval_jit(batch).reshape((-1,))
    expected = jnp.array([0.3, 1.7], dtype=float)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_pipeline_points_vs_coord_separable():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    anchors = {"x": jnp.array([[0.25]], dtype=float)}
    values = jnp.array([0.0625], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=values)

    pipes = EnforcedConstraintPipelines.build(
        functions={"u": u}, interior_data=[interior]
    )
    u_enforced = pipes.apply({"u": u})["u"]
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)

    component = geom.component()
    sep = component.sample_coord_separable({"x": 6}, num_points=())
    x_axis = sep.points["x"][0].data
    dense = _line_batch(geom, xs=x_axis)

    out_sep = eval_jit(u_enforced, sep).reshape((-1,))
    out_dense = eval_jit(u_enforced, dense).reshape((-1,))
    assert jnp.allclose(out_sep, out_dense, atol=1e-6)


def test_operator_stack_with_pipeline():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    anchors = {"x": jnp.array([[0.25]], dtype=float)}
    values = jnp.array([0.0625], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=values)

    pipes = EnforcedConstraintPipelines.build(
        functions={"u": u}, interior_data=[interior]
    )
    u_enforced = pipes.apply({"u": u})["u"]

    component = geom.component()
    batch = component.sample(
        256, structure=ProductStructure((("x",),)), sampler="latin_hypercube"
    )
    du = partial_x(u_enforced)

    eval_jit = eqx.filter_jit(lambda: mean(du, batch, component=component).data)
    out = eval_jit()
    assert jnp.allclose(out, 1.0, atol=0.2)


def test_jit_pipeline_determinism():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] + 1.0

    anchors = {"x": jnp.array([[0.5]], dtype=float)}
    values = jnp.array([1.5], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=values)

    pipes = EnforcedConstraintPipelines.build(
        functions={"u": u}, interior_data=[interior]
    )
    u_enforced = pipes.apply({"u": u})["u"]

    batch = _line_batch(geom, xs=jnp.array([0.25, 0.75], dtype=float))
    eval_jit = eqx.filter_jit(lambda b: u_enforced(b).data)
    out1 = eval_jit(batch).reshape((-1,))
    out2 = eval_jit(batch).reshape((-1,))
    assert jnp.allclose(out1, out2, atol=1e-8)


def test_pipeline_passthrough_for_unconstrained_field():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] * 0.0

    @geom.Function("x")
    def v(x):
        return x[0] * 2.0

    boundary = geom.component({"x": Boundary()})
    constraint = SingleFieldEnforcedConstraint(
        "u",
        boundary,
        lambda f: enforce_dirichlet(f, boundary, var="x", target=0.0),
    )

    pipes = EnforcedConstraintPipelines.build(
        functions={"u": u, "v": v},
        constraints=[constraint],
        num_reference=128,
    )
    enforced = pipes.apply({"u": u, "v": v})

    batch = _line_batch(geom, xs=jnp.array([0.2, 0.8], dtype=float))
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)
    out_v = eval_jit(enforced["v"], batch).reshape((-1,))
    expected = jnp.array([0.4, 1.6], dtype=float)
    assert jnp.allclose(out_v, expected, atol=1e-6)


def test_boundary_constraint_multiple_geometry_labels_error():
    geom = Interval1d(0.0, 1.0)
    momentum = Interval1d(-1.0, 1.0).relabel("p")
    domain = geom @ momentum

    @domain.Function("x", "p")
    def u(x, p):
        return x[0] + p[0]

    both = domain.component({"x": Boundary(), "p": Boundary()})
    constraint = SingleFieldEnforcedConstraint(
        "u",
        both,
        lambda f: enforce_dirichlet(f, both, var="x", target=1.0),
    )

    with pytest.raises(ValueError, match="exactly one geometry Boundary"):
        EnforcedConstraintPipelines.build(functions={"u": u}, constraints=[constraint])


def test_sensor_tracks_require_xt_domain():
    geom = Interval1d(0.0, 1.0)
    momentum = Interval1d(-1.0, 1.0).relabel("p")
    time = TimeInterval(0.0, 1.0)
    domain = geom @ momentum @ time

    @domain.Function("x", "p", "t")
    def u(x, p, t):
        return x[0] + p[0] + t

    interior = EnforcedInteriorData(
        "u",
        sensors=jnp.array([[0.25]], dtype=float),
        times=jnp.array([0.5], dtype=float),
        sensor_values=jnp.array([[1.0]], dtype=float),
    )

    with pytest.raises(ValueError, match="domain labels exactly"):
        EnforcedConstraintPipelines.build(functions={"u": u}, interior_data=[interior])


def test_vector_output_interior_data():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.stack([x[0], x[0] * 2.0], axis=-1)

    anchors = {"x": jnp.array([[0.25], [0.75]], dtype=float)}
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=values)

    pipes = EnforcedConstraintPipelines.build(
        functions={"u": u}, interior_data=[interior]
    )
    u_enforced = pipes.apply({"u": u})["u"]

    batch = _line_batch(geom, xs=anchors["x"])
    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)
    out = eval_jit(u_enforced, batch).reshape(values.shape)
    assert jnp.allclose(out, values, atol=1e-4)


def test_where_all_weight_all_mean():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0]

    @geom.Function("x")
    def mask(x):
        return x[0] > 0.5

    @geom.Function("x")
    def weight(x):
        return x[0] * 0.0 + 2.0

    component = geom.component(where_all=mask, weight_all=weight)
    batch = _line_batch(geom, xs=jnp.array([0.25, 0.75], dtype=float))
    eval_jit = eqx.filter_jit(lambda: mean(u, batch, component=component).data)
    out = eval_jit().reshape(())
    assert jnp.allclose(out, 0.75, atol=1e-6)

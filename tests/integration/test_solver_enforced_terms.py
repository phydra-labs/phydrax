#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import warnings

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax._frozendict import frozendict
from phydrax.constraints import (
    ContinuousInitialConstraint,
    ContinuousPointwiseInteriorConstraint,
    enforce_dirichlet,
)
from phydrax.domain import (
    Boundary,
    FixedStart,
    FourierAxisSpec,
    Interval1d,
    PointsBatch,
    ProductStructure,
    TimeInterval,
)
from phydrax.domain._structure import CoordSeparableBatch
from phydrax.nn.models import LatentContractionModel
from phydrax.nn.models.core._base import _AbstractBaseModel
from phydrax.operators.differential import dt, dt_n, laplacian
from phydrax.operators.differential._hooks import get_derivative_hook
from phydrax.solver import (
    EnforcedInteriorData,
    FunctionalSolver,
    SingleFieldEnforcedConstraint,
)


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


def test_functional_solver_builds_enforced_pipeline_terms():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + t

    boundary_component = domain.component({"x": Boundary()})
    initial_component = domain.component({"t": FixedStart()})

    boundary_constraint = SingleFieldEnforcedConstraint(
        "u",
        boundary_component,
        lambda f: enforce_dirichlet(f, boundary_component, var="x", target=5.0),
    )
    initial_constraint = SingleFieldEnforcedConstraint(
        "u",
        initial_component,
        lambda f: enforce_dirichlet(f, initial_component, var="t", target=2.0),
    )

    anchors = {
        "x": jnp.array([[0.25], [0.75]], dtype=float),
        "t": jnp.array([0.6, 0.4], dtype=float),
    }
    anchor_values = jnp.array([3.0, 4.0], dtype=float)
    interior = EnforcedInteriorData(
        "u",
        points=anchors,
        values=anchor_values,
        eps_snap=1e-12,
    )

    solver = FunctionalSolver(
        functions={"u": u},
        constraints=(),
        constraint_terms=[boundary_constraint, initial_constraint],
        interior_data_terms=[interior],
        boundary_weight_num_reference=256,
    )
    u_enforced = solver.ansatz_functions()["u"]
    eval_jit = eqx.filter_jit(lambda b: u_enforced(b).data)

    out = eval_jit(
        _paired_batch(domain, xs=jnp.array([0.0, 1.0]), ts=jnp.array([0.5, 0.5]))
    )
    assert jnp.allclose(out.reshape((-1,)), 5.0, atol=1e-3)

    out = eval_jit(
        _paired_batch(domain, xs=jnp.array([0.5, 0.5]), ts=jnp.array([0.0, 0.0]))
    )
    assert jnp.allclose(out.reshape((-1,)), 2.0, atol=1e-3)

    out = eval_jit(
        _paired_batch(domain, xs=jnp.array([0.25, 0.75]), ts=jnp.array([0.6, 0.4]))
    )
    assert jnp.allclose(out.reshape((-1,)), anchor_values, atol=1e-3)


def test_initial_constraint_coord_separable_spatial():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return 0.0

    initial_component = domain.component({"t": FixedStart()})
    structure = ProductStructure((("x",),))

    constraint = ContinuousInitialConstraint(
        "u",
        initial_component,
        func=0.0,
        num_points={"x": 5},
        structure=structure,
    )

    batch = constraint.sample(key=jr.key(0))
    assert isinstance(batch, CoordSeparableBatch)
    assert isinstance(batch.points["x"], tuple)
    assert len(batch.points["x"]) == 1

    loss = constraint.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0, atol=1e-6)


def _as_scalar(x):
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return arr
    if arr.size != 1:
        raise ValueError("Expected scalar input for scalar factor model.")
    return arr.reshape(())


class _ScalarLatentModel(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str

    def __init__(self) -> None:
        self.in_size = "scalar"
        self.out_size = 2

    def __call__(self, x, /, *, key=jr.key(0)):
        x = _as_scalar(x)
        return jnp.stack([x, jnp.array(1.0)], axis=-1)


def test_enforced_constraints_respected_with_latent_contraction_model():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=_ScalarLatentModel(),
        t=_ScalarLatentModel(),
    )
    u_raw = domain.Model("x", "t")(model)

    boundary = domain.component({"x": Boundary()})
    initial = domain.component({"t": FixedStart()})

    @domain.Function("x")
    def u0_target(x):
        return x[0]

    @domain.Function("x")
    def ut0_target(x):
        del x
        return 0.0

    terms = [
        SingleFieldEnforcedConstraint(
            "u",
            boundary,
            lambda f: enforce_dirichlet(f, boundary, var="x", target=0.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=0,
            initial_target=u0_target,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=1,
            initial_target=ut0_target,
        ),
    ]

    solver = FunctionalSolver(
        functions={"u": u_raw},
        constraints=(),
        constraint_terms=terms,
        boundary_weight_num_reference=128,
    )
    u = solver.ansatz_functions()["u"]

    t_vals = jnp.linspace(0.0, 1.0, 17)
    left = jax.vmap(lambda t: u.func(jnp.asarray([0.0]), t))(t_vals).reshape((-1,))
    right = jax.vmap(lambda t: u.func(jnp.asarray([1.0]), t))(t_vals).reshape((-1,))
    assert jnp.max(jnp.abs(left)) < 1e-6
    assert jnp.max(jnp.abs(right)) < 1e-6

    x_vals = jnp.linspace(0.0, 1.0, 17)
    x_interior = x_vals[1:-1]
    u_init = jax.vmap(lambda x: u.func(jnp.asarray([x]), 0.0))(x_interior).reshape((-1,))
    assert jnp.max(jnp.abs(u_init - x_interior)) < 1e-6

    corner = u.func(jnp.asarray([1.0]), 0.0)
    assert jnp.abs(corner) < 1e-6


def test_enforced_initial_overlay_boundary_compatible_preserves_all_orders():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=_ScalarLatentModel(),
        t=_ScalarLatentModel(),
    )
    u_raw = domain.Model("x", "t")(model)

    boundary = domain.component({"x": Boundary()})
    initial = domain.component({"t": FixedStart()})

    @domain.Function("x")
    def u0_target(x):
        return jnp.sin(jnp.pi * x[0])

    @domain.Function("x")
    def ut0_target(x):
        return jnp.sin(jnp.pi * x[0])

    @domain.Function("x")
    def utt0_target(x):
        return -jnp.sin(jnp.pi * x[0])

    terms = [
        SingleFieldEnforcedConstraint(
            "u",
            boundary,
            lambda f: enforce_dirichlet(f, boundary, var="x", target=0.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=0,
            initial_target=u0_target,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=1,
            initial_target=ut0_target,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=2,
            initial_target=utt0_target,
        ),
    ]

    solver = FunctionalSolver(
        functions={"u": u_raw},
        constraints=(),
        constraint_terms=terms,
        boundary_weight_num_reference=128,
    )
    u = solver.ansatz_functions()["u"]

    t_vals = jnp.linspace(0.0, 1.0, 33)
    left = jax.vmap(lambda t: u.func(jnp.asarray([0.0]), t))(t_vals).reshape((-1,))
    right = jax.vmap(lambda t: u.func(jnp.asarray([1.0]), t))(t_vals).reshape((-1,))
    assert jnp.max(jnp.abs(left)) < 1e-6
    assert jnp.max(jnp.abs(right)) < 1e-6

    x_vals = jnp.linspace(0.0, 1.0, 33)
    expected = jnp.sin(jnp.pi * x_vals)

    u_init = jax.vmap(lambda x: u.func(jnp.asarray([x]), 0.0))(x_vals).reshape((-1,))
    assert jnp.max(jnp.abs(u_init - expected)) < 1e-6

    ut = dt(u, var="t")
    ut_init = jax.vmap(lambda x: ut.func(jnp.asarray([x]), 0.0))(x_vals).reshape((-1,))
    assert jnp.max(jnp.abs(ut_init - expected)) < 1e-6

    utt = dt_n(u, var="t", order=2)
    utt_init = jax.vmap(lambda x: utt.func(jnp.asarray([x]), 0.0))(x_vals).reshape((-1,))
    assert jnp.max(jnp.abs(utt_init + expected)) < 1e-6


def test_enforced_initial_overlay_incompatible_keeps_boundary_gate():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=_ScalarLatentModel(),
        t=_ScalarLatentModel(),
    )
    u_raw = domain.Model("x", "t")(model)

    boundary = domain.component({"x": Boundary()})
    initial = domain.component({"t": FixedStart()})

    @domain.Function("x")
    def u0_target(x):
        return x[0]

    @domain.Function("x")
    def ut0_target(x):
        return 1.0 + x[0]

    terms = [
        SingleFieldEnforcedConstraint(
            "u",
            boundary,
            lambda f: enforce_dirichlet(f, boundary, var="x", target=0.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=0,
            initial_target=u0_target,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=1,
            initial_target=ut0_target,
        ),
    ]

    solver = FunctionalSolver(
        functions={"u": u_raw},
        constraints=(),
        constraint_terms=terms,
        boundary_weight_num_reference=128,
    )
    assert solver.constraint_pipelines is not None
    pipeline = solver.constraint_pipelines.pipelines["u"]
    assert pipeline.initial_overlay_boundary_compatible is False

    u = solver.ansatz_functions()["u"]
    boundary_corner = u.func(jnp.asarray([1.0]), 0.0)
    assert jnp.abs(boundary_corner) < 1e-6

    target_corner = u0_target.func(jnp.asarray([1.0]))
    assert jnp.abs(boundary_corner - target_corner) > 1e-2

    ut = dt(u, var="t")
    ut_corner = ut.func(jnp.asarray([1.0]), 0.0)
    target_ut_corner = ut0_target.func(jnp.asarray([1.0]))
    assert jnp.abs(ut_corner - target_ut_corner) > 1e-2


def test_wave_like_loss_with_coord_separable_enforced_terms_runs():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=_ScalarLatentModel(),
        t=_ScalarLatentModel(),
    )
    u_raw = domain.Model("x", "t")(model)

    boundary = domain.component({"x": Boundary()})
    initial = domain.component({"t": FixedStart()})

    @domain.Function("x")
    def u0_target(x):
        return x[0]

    @domain.Function("x")
    def ut0_target(x):
        return 0.0 * x[0]

    terms = [
        SingleFieldEnforcedConstraint(
            "u",
            boundary,
            lambda f: enforce_dirichlet(f, boundary, var="x", target=0.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=0,
            initial_target=u0_target,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=1,
            initial_target=ut0_target,
        ),
    ]

    pde = ContinuousPointwiseInteriorConstraint(
        "u",
        domain,
        operator=lambda f: dt_n(f, var="t", order=2) - laplacian(f, var="x"),
        num_points={"x": 16, "t": 8},
        structure=ProductStructure((("x",), ("t",))),
        reduction="mean",
    )

    solver = FunctionalSolver(
        functions={"u": u_raw},
        constraints=[pde],
        constraint_terms=terms,
        boundary_weight_num_reference=128,
    )

    loss = solver.loss(key=jr.key(9))
    assert jnp.isfinite(loss)


def test_enforced_pipeline_stacked_overlays_keep_derivative_hooks_and_fast_path():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=_ScalarLatentModel(),
        t=_ScalarLatentModel(),
    )
    u_raw = domain.Model("x", "t")(model)

    boundary = domain.component({"x": Boundary()})
    initial = domain.component({"t": FixedStart()})

    @domain.Function("x")
    def u0_target(x):
        return x[0]

    @domain.Function("x")
    def ut0_target(x):
        return 0.0 * x[0]

    terms = [
        SingleFieldEnforcedConstraint(
            "u",
            boundary,
            lambda f: enforce_dirichlet(f, boundary, var="x", target=0.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=0,
            initial_target=u0_target,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: f,
            time_derivative_order=1,
            initial_target=ut0_target,
        ),
    ]

    solver = FunctionalSolver(
        functions={"u": u_raw},
        constraints=(),
        constraint_terms=terms,
        boundary_weight_num_reference=128,
    )
    u = solver.ansatz_functions()["u"]
    assert get_derivative_hook(u) is not None

    du_dt = dt_n(u, var="t", order=1, backend="ad")
    batch = domain.component().sample_coord_separable(
        {"x": FourierAxisSpec(10)},
        num_points=6,
        dense_structure=ProductStructure((("t",),)),
        key=jr.key(13),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = du_dt(batch)

    fallback_msgs = [
        str(w.message)
        for w in caught
        if "Falling back to generic derivative path for LatentContractionModel"
        in str(w.message)
    ]
    assert not fallback_msgs
    assert jnp.all(jnp.isfinite(jnp.asarray(out.data)))

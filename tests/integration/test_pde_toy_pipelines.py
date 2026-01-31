#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.constraints import enforce_dirichlet
from phydrax.constraints._continuous_interior import (
    ContinuousInitialFunctionConstraint,
    ContinuousPointwiseInteriorConstraint,
)
from phydrax.domain import (
    Boundary,
    FixedStart,
    FourierAxisSpec,
    Interval1d,
    ProductStructure,
    TimeInterval,
)
from phydrax.operators.differential import bilaplacian, dt, laplacian
from phydrax.solver import (
    EnforcedInteriorData,
    FunctionalSolver,
    SingleFieldEnforcedConstraint,
)


def test_pde_toy_steady_pipeline_zero_loss():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return 1.0

    left = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    right = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] >= 0.5})

    left_constraint = SingleFieldEnforcedConstraint(
        "u",
        left,
        lambda f: enforce_dirichlet(f, left, var="x", target=1.0),
    )
    right_constraint = SingleFieldEnforcedConstraint(
        "u",
        right,
        lambda f: enforce_dirichlet(f, right, var="x", target=1.0),
    )

    anchors = {"x": jnp.array([[0.25], [0.75]], dtype=float)}
    values = jnp.array([1.0, 1.0], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=values)

    structure = ProductStructure((("x",),))
    pde_constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: laplacian(f, var="x"),
        num_points=64,
        structure=structure,
    )

    solver = FunctionalSolver(
        functions={"u": u},
        constraints=[pde_constraint],
        constraint_terms=[left_constraint, right_constraint],
        interior_data_terms=[interior],
        boundary_weight_num_reference=256,
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_steady_pipeline_zero_loss_jet_backend():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return 1.0

    # Jet cannot be mixed with enforced boundary constraints (Boundary() enforced constraints /
    # EnforcedInteriorData) because the enforced pipeline traces through the MLS/BVH weight
    # computation, which uses primitives not supported by jax.experimental.jet
    # (e.g. lax.cond, softplus/logaddexp custom_jvp, and clip/min/max rules).
    structure = ProductStructure((("x",),))
    pde_constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: laplacian(f, var="x", backend="jet"),
        num_points=64,
        structure=structure,
    )

    solver = FunctionalSolver(
        functions={"u": u},
        constraints=[pde_constraint],
        constraint_terms=[],
        interior_data_terms=[],
        boundary_weight_num_reference=256,
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_steady_pipeline_zero_loss_basis_backend_coord_separable():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return 1.0

    left = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    right = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] >= 0.5})

    left_constraint = SingleFieldEnforcedConstraint(
        "u",
        left,
        lambda f: enforce_dirichlet(f, left, var="x", target=1.0),
    )
    right_constraint = SingleFieldEnforcedConstraint(
        "u",
        right,
        lambda f: enforce_dirichlet(f, right, var="x", target=1.0),
    )

    anchors = {"x": jnp.array([[0.25], [0.75]], dtype=float)}
    values = jnp.array([1.0, 1.0], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=values)

    structure = ProductStructure((("x",),))
    pde_constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: laplacian(
            f,
            var="x",
            backend="basis",
            basis="fourier",
            periodic=True,
        ),
        num_points=0,
        structure=structure,
        coord_separable={"x": FourierAxisSpec(64)},
    )

    solver = FunctionalSolver(
        functions={"u": u},
        constraints=[pde_constraint],
        constraint_terms=[left_constraint, right_constraint],
        interior_data_terms=[interior],
        boundary_weight_num_reference=256,
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_steady_pipeline_zero_loss_bilaplacian_jet_backend():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return 1.0

    # Jet cannot be mixed with enforced boundary constraints (Boundary() enforced constraints /
    # EnforcedInteriorData) because the enforced pipeline traces through the MLS/BVH weight
    # computation, which uses primitives not supported by jax.experimental.jet.
    structure = ProductStructure((("x",),))
    pde_constraint = ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: bilaplacian(f, var="x", backend="jet"),
        num_points=64,
        structure=structure,
    )

    solver = FunctionalSolver(
        functions={"u": u},
        constraints=[pde_constraint],
        constraint_terms=[],
        interior_data_terms=[],
        boundary_weight_num_reference=256,
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_transient_pipeline_zero_loss():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    left = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    right = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] >= 0.5})
    initial = domain.component({"t": FixedStart()})

    constraints = [
        SingleFieldEnforcedConstraint(
            "u",
            left,
            lambda f: enforce_dirichlet(f, left, var="x", target=1.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            right,
            lambda f: enforce_dirichlet(f, right, var="x", target=1.0),
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=1.0),
            time_derivative_order=0,
            initial_target=1.0,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=1.0),
            time_derivative_order=1,
            initial_target=0.0,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=1.0),
            time_derivative_order=2,
            initial_target=0.0,
        ),
    ]

    anchors = {
        "x": jnp.array([[0.25], [0.75]], dtype=float),
        "t": jnp.array([0.4, 0.6], dtype=float),
    }
    values = jnp.array([1.0, 1.0], dtype=float)
    interior = EnforcedInteriorData("u", points=anchors, values=values)

    structure = ProductStructure((("x", "t"),))
    pde_time = ContinuousPointwiseInteriorConstraint(
        "u",
        domain,
        operator=lambda f: dt(f, var="t"),
        num_points=64,
        structure=structure,
    )
    pde_space = ContinuousPointwiseInteriorConstraint(
        "u",
        domain,
        operator=lambda f: laplacian(f, var="x"),
        num_points=64,
        structure=structure,
    )

    solver = FunctionalSolver(
        functions={"u": u},
        constraints=[pde_time, pde_space],
        constraint_terms=constraints,
        interior_data_terms=[interior],
        boundary_weight_num_reference=256,
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_transient_pipeline_zero_loss_jet_backend():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    # Jet cannot be mixed with enforced boundary constraints (Boundary() enforced constraints /
    # EnforcedInteriorData) because the enforced pipeline traces through the MLS/BVH weight
    # computation, which uses primitives not supported by jax.experimental.jet.
    structure = ProductStructure((("x", "t"),))
    pde_time = ContinuousPointwiseInteriorConstraint(
        "u",
        domain,
        operator=lambda f: dt(f, var="t"),
        num_points=64,
        structure=structure,
    )
    pde_space = ContinuousPointwiseInteriorConstraint(
        "u",
        domain,
        operator=lambda f: laplacian(f, var="x", backend="jet"),
        num_points=64,
        structure=structure,
    )

    solver = FunctionalSolver(
        functions={"u": u},
        constraints=[pde_time, pde_space],
        constraint_terms=[],
        interior_data_terms=[],
        boundary_weight_num_reference=256,
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_transient_enforced_initial_targets_dt2_zero_jet_backend():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0 + t**2

    initial = domain.component({"t": FixedStart()})
    enforced_initial_constraints = [
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=1.0),
            time_derivative_order=0,
            initial_target=1.0,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=1.0),
            time_derivative_order=1,
            initial_target=0.0,
        ),
        SingleFieldEnforcedConstraint(
            "u",
            initial,
            lambda f: enforce_dirichlet(f, initial, var="t", target=1.0),
            time_derivative_order=2,
            initial_target=0.0,
        ),
    ]

    constraint = ContinuousInitialFunctionConstraint(
        "u",
        domain,
        func=0.0,
        time_derivative_order=2,
        time_derivative_backend="jet",
        num_points=64,
        structure=ProductStructure((("x",),)),
    )

    solver = FunctionalSolver(
        functions={"u": u},
        constraints=[constraint],
        constraint_terms=enforced_initial_constraints,
        interior_data_terms=[],
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import (
    Boundary,
    FixedStart,
    Interval1d,
    PointBatch,
    SampleLayout,
    TimeInterval,
)
from phydrax.enforcement import (
    enforce_dirichlet,
    EnforcementProgram,
    EnforcementSpec,
)
from phydrax.solver import FunctionalSolver


def _line_batch(domain, xs):
    structure = SampleLayout((("x",),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {"x": cx.Field(jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None))}
    )
    return PointBatch(points=points, structure=structure)


def test_boundary_subset_blend_matches_pieces():
    geom = Interval1d(0.0, 1.0)

    def left_where(x):
        return x[0] < 0.5

    def right_where(x):
        return x[0] >= 0.5

    @geom.Function("x")
    def u(x):
        return x[0] * 0.0

    left_component = geom.component({"x": Boundary()}, where={"x": left_where})
    right_component = geom.component({"x": Boundary()}, where={"x": right_where})
    full_boundary = geom.component({"x": Boundary()})

    left_constraint = EnforcementSpec(
        phx.conditions.Dirichlet("u", left_component, target=1.0),
        kind="custom",
        transform=lambda f, _: enforce_dirichlet(f, full_boundary, var="x", target=1.0),
    )
    right_constraint = EnforcementSpec(
        phx.conditions.Dirichlet("u", right_component, target=2.0),
        kind="custom",
        transform=lambda f, _: enforce_dirichlet(f, full_boundary, var="x", target=2.0),
    )

    pipelines = EnforcementProgram.build(
        functions={"u": u},
        specs=[left_constraint, right_constraint],
        num_reference=256,
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    batch = _line_batch(geom, xs=jnp.array([0.0, 1.0]))
    out = jnp.asarray(u_enforced(batch).data).reshape((-1,))
    assert jnp.allclose(out[0], 1.0, atol=1e-3)
    assert jnp.allclose(out[1], 2.0, atol=1e-3)


def test_initial_overlay_boundary_gate_is_dimensionless_and_scale_invariant():
    gate_values = []

    for length in (1.0, 100.0):
        domain = Interval1d(0.0, length) @ TimeInterval(0.0, 1.0)
        boundary = domain.component({"x": Boundary()})
        initial = domain.component({"t": FixedStart()})

        @domain.Function("x", "t")
        def u(x, t):
            return x[0] * 0.0 + t * 0.0

        @domain.Function("x")
        def initial_target(x):
            return jnp.sin(jnp.pi * x[0] / length)

        specs = [
            EnforcementSpec(phx.conditions.Dirichlet("u", boundary, target=0.0)),
            EnforcementSpec(
                phx.conditions.Initial("u", initial, target=initial_target, order=0)
            ),
        ]
        pipelines = EnforcementProgram.build(
            functions={"u": u},
            specs=specs,
            num_reference=64,
        )
        gate = pipelines.pipelines["u"].boundary_gate
        assert gate is not None
        gate_values.append(
            jnp.stack(
                (
                    gate.func(jnp.array([0.25 * length])),
                    gate.func(jnp.array([0.5 * length])),
                )
            )
        )

    assert jnp.allclose(gate_values[0], jnp.array([0.75, 1.0]))
    assert jnp.allclose(gate_values[0], gate_values[1])


def test_initial_overlay_gate_preserves_declared_normal_derivative():
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    ) @ TimeInterval(0.0, 1.0)
    boundary = domain.component({"x": Boundary()})
    initial = domain.component({"t": FixedStart()})

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] * 0.0 + t * 0.0

    @domain.Function("x")
    def incompatible_initial_target(x):
        return x[0]

    specs = [
        EnforcementSpec(
            phx.conditions.Neumann("u", boundary, var="x", target=0.0, mode="forward")
        ),
        EnforcementSpec(
            phx.conditions.Initial(
                "u", initial, target=incompatible_initial_target, order=0
            )
        ),
    ]
    pipelines = EnforcementProgram.build(
        functions={"u": u},
        specs=specs,
        num_reference=64,
    )
    pipeline = pipelines.pipelines["u"]
    assert pipeline.initial_overlay_boundary_compatible is False

    u_enforced = pipelines.apply({"u": u})["u"]
    point = jnp.array([1.0, 0.0])
    normal = boundary.normal(var="x").func(point)
    derivative = jnp.dot(
        jax.grad(lambda x: u_enforced.func(x, jnp.array(0.0)))(point),
        normal,
    )

    assert jnp.allclose(derivative, 0.0, atol=1e-10)


def test_functional_solver_configures_cad_preservation_gate_extent():
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    ) @ TimeInterval(0.0, 1.0)
    boundary = domain.component({"x": Boundary()})
    initial = domain.component({"t": FixedStart()})

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] * 0.0 + t * 0.0

    @domain.Function("x")
    def initial_target(x):
        return (1.0 - x[0] ** 2) * (1.0 - x[1] ** 2)

    specs = [
        EnforcementSpec(phx.conditions.Dirichlet("u", boundary, target=0.0)),
        EnforcementSpec(
            phx.conditions.Initial("u", initial, target=initial_target, order=0)
        ),
    ]
    functions = {"u": u}
    values = []
    for saturation_fraction in (0.5, 0.2):
        program = phx.enforcement.compile(
            functions,
            specs,
            options=phx.enforcement.EnforcementOptions(
                gate_method="compact",
                gate_saturation_fraction=saturation_fraction,
                num_reference=64,
            ),
        )
        solver = FunctionalSolver(
            functions=functions,
            terms=(),
            enforcement=program,
        )
        assert solver.enforcement is not None
        gate = solver.enforcement.pipelines["u"].boundary_gate
        assert gate is not None
        values.append(gate.func(jnp.array([0.8, 0.0])))

    assert 0.2 < float(values[0]) < 0.4
    assert 0.6 < float(values[1]) < 0.8

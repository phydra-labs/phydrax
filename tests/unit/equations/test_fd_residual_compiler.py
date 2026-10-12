#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.equations._ir import depends_on_fields, field_polynomial_degree


pytestmark = pytest.mark.strict_jax


def _grid(shape: tuple[int, ...]) -> phx.discretization.PreparedTensorGrid:
    names = ("x", "y")[: len(shape)]
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(size) for size in shape),
        axis_names=names,
    ).prepare(jnp.asarray([[0.0] * len(shape), [1.0] * len(shape)], dtype=jnp.float64))


def _poisson_problem(*, nonlinear: bool = False) -> phx.equations.PDEProblemIR:
    u = phx.equations.PDEExpression.field("u")
    lhs = u.derivative("x", order=2) + u.derivative("y", order=2)
    if nonlinear:
        lhs = lhs + u * u
    return phx.equations.PDEProblemIR(
        coordinates=tuple(
            phx.equations.PDECoordinate(axis, "space", bounds=(0.0, 1.0))
            for axis in ("x", "y")
        ),
        fields=(phx.equations.PDEField("u", coordinates=("x", "y")),),
        parameters=tuple(
            phx.equations.PDEParameter(name) for name in ("forcing", "boundary")
        ),
        equations=(
            phx.equations.PDEEquation(
                "poisson", lhs, phx.equations.PDEExpression.parameter("forcing")
            ),
        ),
        regions=tuple(
            phx.equations.PDERegion(axis + "-edge", "boundary", (axis,))
            for axis in ("x", "y")
        ),
        conditions=tuple(
            phx.equations.PDECondition(
                axis + "-dirichlet",
                "boundary",
                u,
                target=phx.equations.PDEExpression.parameter("boundary"),
                region=axis + "-edge",
                coordinate=axis,
            )
            for axis in ("x", "y")
        ),
    )


def _five_point(values: np.ndarray, /) -> np.ndarray:
    hx = 1.0 / (values.shape[0] - 1)
    hy = 1.0 / (values.shape[1] - 1)
    return (values[2:, 1:-1] - 2.0 * values[1:-1, 1:-1] + values[:-2, 1:-1]) / hx**2 + (
        values[1:-1, 2:] - 2.0 * values[1:-1, 1:-1] + values[1:-1, :-2]
    ) / hy**2


def test_five_point_poisson_and_runtime_dirichlet_elimination() -> None:
    shape = (7, 6)
    compiled = phx.equations.compile_finite_difference_residual(
        _poisson_problem(), _grid(shape), equation_targets={"poisson": "u"}
    )
    state = jnp.arange((shape[0] - 2) * (shape[1] - 2), dtype=jnp.float64) / 19.0
    forcing = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 37.0
    for boundary in (0.0, 1.25):
        args = {
            "forcing": jnp.asarray(forcing),
            "boundary": jnp.asarray(boundary, dtype=jnp.float64),
        }
        expected = np.full(shape, boundary, dtype=np.float64)
        expected[1:-1, 1:-1] = np.asarray(state).reshape((shape[0] - 2, shape[1] - 2))
        np.testing.assert_allclose(compiled.expand(state, args)["u"], expected)
        residual = eqx.filter_jit(compiled.residual)(state, args)
        np.testing.assert_allclose(
            residual, (_five_point(expected) - forcing[1:-1, 1:-1]).ravel(), atol=1e-12
        )
    assert compiled.state_space.compatible(compiled.residual_space)
    assert compiled.state_space.shape == (20,)


def test_affine_bubble_system_and_linearization_duality() -> None:
    shape = (7, 8)
    grid = _grid(shape)
    compiled = phx.equations.compile_finite_difference_residual(
        _poisson_problem(), grid, equation_targets={"poisson": "u"}
    )
    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0]), np.linspace(0.0, 1.0, shape[1]), indexing="ij"
    )
    exact = x * (x - 1.0) * y * (y - 1.0)
    forcing = 2.0 * (x * (x - 1.0) + y * (y - 1.0))
    args = {"forcing": jnp.asarray(forcing), "boundary": jnp.zeros((), dtype=jnp.float64)}
    system, rhs = compiled.linear_system(args)
    dense = np.asarray(
        phx.linalg.materialize(
            system.operator, phx.linalg.MaterializationPolicy(max_entries=900)
        )
    )
    solved = np.linalg.solve(dense, np.asarray(rhs))
    np.testing.assert_allclose(solved, exact[1:-1, 1:-1].ravel(), atol=1e-13)
    np.testing.assert_allclose(
        compiled.residual(jnp.asarray(solved), args), 0.0, atol=1e-12
    )
    point = jnp.linspace(-0.4, 0.6, compiled.state_space.size, dtype=jnp.float64)
    operator = compiled.linearization(point, args)
    v = jnp.sin(jnp.arange(point.size, dtype=jnp.float64))
    w = jnp.cos(jnp.arange(point.size, dtype=jnp.float64))
    np.testing.assert_allclose(
        jnp.vdot(operator.mv(v), w), jnp.vdot(v, operator.transpose_mv(w)), atol=1e-11
    )


@pytest.mark.parametrize(
    "kind", ["neumann", "robin"], ids=["prescribed-derivative", "affine-robin"]
)
def test_nondirichlet_nodes_keep_runtime_derivative_rows(kind: str) -> None:
    u = phx.equations.PDEExpression.field("u")
    derivative = u.derivative("x")
    expression = derivative if kind == "neumann" else 2.0 * u + 3.0 * derivative
    problem = phx.equations.PDEProblemIR(
        coordinates=(phx.equations.PDECoordinate("x", "space"),),
        fields=(phx.equations.PDEField("u", coordinates=("x",)),),
        parameters=(phx.equations.PDEParameter("flux"),),
        equations=(
            phx.equations.PDEEquation(
                "gradient", derivative, phx.equations.PDEExpression.constant(0.0)
            ),
        ),
        regions=(phx.equations.PDERegion("edge", "boundary", ("x",)),),
        conditions=(
            phx.equations.PDECondition(
                "law",
                "boundary",
                expression,
                target=phx.equations.PDEExpression.parameter("flux"),
                region="edge",
                coordinate="x",
            ),
        ),
    )
    compiled = phx.equations.compile_finite_difference_residual(
        problem, _grid((7,)), equation_targets={"gradient": "u"}
    )
    state = jnp.linspace(0.0, 1.0, 7, dtype=jnp.float64) ** 2
    args = {"flux": jnp.asarray(0.75, dtype=jnp.float64)}
    result = np.asarray(compiled.residual(state, args))
    np.testing.assert_allclose(
        result[1:-1], 2.0 * np.linspace(0.0, 1.0, 7)[1:-1], atol=1e-12
    )
    endpoints = np.asarray(state)[[0, -1]]
    expected = (
        np.full((2,), 0.75) if kind == "neumann" else (0.75 - 2.0 * endpoints) / 3.0
    )
    np.testing.assert_allclose(result[[0, -1]], expected)
    assert compiled.state_space.shape == (7,)
    system, rhs = compiled.linear_system(args)
    np.testing.assert_allclose(system.operator.mv(state) - rhs, result, atol=1e-12)


def test_coupled_fields_use_distinct_target_masks_and_canonical_mapping() -> None:
    u, v = (phx.equations.PDEExpression.field(name) for name in ("u", "v"))
    problem = phx.equations.PDEProblemIR(
        coordinates=(phx.equations.PDECoordinate("x", "space"),),
        fields=tuple(
            phx.equations.PDEField(name, coordinates=("x",)) for name in ("u", "v")
        ),
        equations=(
            phx.equations.PDEEquation("v-balance", v.derivative("x") + 2.0 * u),
            phx.equations.PDEEquation("u-balance", u.derivative("x") + v),
        ),
        regions=(phx.equations.PDERegion("edge", "boundary", ("x",)),),
        conditions=(
            phx.equations.PDECondition(
                "u-edge",
                "boundary",
                u,
                target=phx.equations.PDEExpression.constant(2.0),
                region="edge",
                coordinate="x",
            ),
        ),
    )
    grid = _grid((6,))
    targets = {"v-balance": "v", "u-balance": "u"}
    compiled = phx.equations.compile_finite_difference_residual(
        problem, grid, equation_targets=targets
    )
    state = jnp.concatenate(
        (
            jnp.linspace(0.2, 0.8, 4, dtype=jnp.float64),
            jnp.linspace(0.0, 1.0, 6, dtype=jnp.float64),
        )
    )
    expanded = compiled.expand(state, {})
    x = np.linspace(0.0, 1.0, 6)
    u_values = np.asarray(expanded["u"])
    v_values = np.asarray(expanded["v"])
    expected_u = np.gradient(u_values, x, edge_order=2)[1:-1] + v_values[1:-1]
    expected_v = np.gradient(v_values, x, edge_order=2) + 2.0 * u_values
    np.testing.assert_allclose(
        compiled.residual(state, {}), np.concatenate((expected_u, expected_v)), atol=1e-12
    )
    assert compiled.state_space.shape == (10,)
    reordered = phx.equations.compile_finite_difference_residual(
        problem, grid, equation_targets=dict(reversed(tuple(targets.items())))
    )
    assert compiled.compilation_id == reordered.compilation_id


def test_nonlinear_and_field_dependent_boundary_targets_refuse_linear_system() -> None:
    grid = _grid((5, 5))
    nonlinear = phx.equations.compile_finite_difference_residual(
        _poisson_problem(nonlinear=True), grid, equation_targets={"poisson": "u"}
    )
    with pytest.raises(ValueError, match="affine"):
        nonlinear.linear_system({})
    point = jnp.linspace(0.1, 0.9, nonlinear.state_space.size, dtype=jnp.float64)
    args = {
        "forcing": jnp.zeros(grid.shape, dtype=jnp.float64),
        "boundary": jnp.zeros((), dtype=jnp.float64),
    }
    full = np.zeros(grid.shape, dtype=np.float64)
    full[1:-1, 1:-1] = np.asarray(point).reshape((3, 3))
    np.testing.assert_allclose(
        nonlinear.residual(point, args),
        (_five_point(full) + full[1:-1, 1:-1] ** 2).ravel(),
        atol=1e-12,
    )
    operator = nonlinear.linearization(point, args)
    v, w = jnp.sin(point), jnp.cos(point)
    _, independent_jvp = jax.jvp(
        lambda value: nonlinear.residual(value, args), (point,), (v,)
    )
    np.testing.assert_allclose(operator.mv(v), independent_jvp, atol=1e-12)
    np.testing.assert_allclose(
        jnp.vdot(operator.mv(v), w), jnp.vdot(v, operator.transpose_mv(w)), atol=1e-11
    )
    u = phx.equations.PDEExpression.field("u")
    problem = _poisson_problem()
    dependent = replace(
        problem,
        conditions=tuple(
            replace(condition, expression=u.derivative(axis), target=u)
            for condition, axis in zip(problem.conditions, ("x", "y"), strict=True)
        ),
    )
    compiled = phx.equations.compile_finite_difference_residual(
        dependent, grid, equation_targets={"poisson": "u"}
    )
    with pytest.raises(ValueError, match="field-independent"):
        compiled.linear_system({})
    dirichlet = replace(
        problem,
        conditions=tuple(
            replace(condition, target=u) for condition in problem.conditions
        ),
    )
    with pytest.raises(ValueError, match="Dirichlet targets"):
        phx.equations.compile_finite_difference_residual(
            dirichlet, grid, equation_targets={"poisson": "u"}
        )


@pytest.mark.parametrize(
    "case",
    [
        "time",
        "missing",
        "wrong-field",
        "nonbijective",
        "unsupported-boundary",
        "unsupported-target",
        "no-derivative",
        "vector",
    ],
    ids=[
        "time-coordinate",
        "missing-target",
        "unknown-target",
        "duplicate-target",
        "nonlinear-boundary-law",
        "unrepresentable-boundary-target",
        "algebraic-only",
        "component-field",
    ],
)
def test_stationary_compiler_refusals(case: str) -> None:
    problem = _poisson_problem()
    grid = _grid((5, 5))
    targets = {"poisson": "u"}
    match = "bijection"
    u = phx.equations.PDEExpression.field("u")
    if case == "time":
        problem = replace(
            problem,
            coordinates=problem.coordinates + (phx.equations.PDECoordinate("t", "time"),),
        )
        match = "time coordinate"
    elif case == "missing":
        targets = {}
    elif case == "wrong-field":
        targets = {"poisson": "v"}
    elif case == "nonbijective":
        problem = replace(
            problem,
            fields=problem.fields
            + (phx.equations.PDEField("v", coordinates=("x", "y")),),
            equations=problem.equations
            + (phx.equations.PDEEquation("other", u.derivative("x")),),
        )
        targets = {"poisson": "u", "other": "u"}
    elif case == "unsupported-boundary":
        problem = replace(
            problem,
            conditions=tuple(
                replace(condition, expression=u * u) for condition in problem.conditions
            ),
        )
        match = "Unsupported FD boundary"
    elif case == "unsupported-target":
        problem = replace(
            problem,
            conditions=tuple(
                replace(
                    condition,
                    target=phx.equations.PDEExpression.parameter("boundary").derivative(
                        "x"
                    ),
                )
                for condition in problem.conditions
            ),
        )
        match = "Unrepresentable FD boundary target"
    elif case == "no-derivative":
        problem = replace(problem, equations=(phx.equations.PDEEquation("poisson", u),))
        match = "spatial derivative"
    else:
        problem = replace(
            problem,
            fields=(replace(problem.fields[0], representation="vector", components=2),),
            equations=(
                phx.equations.PDEEquation("poisson", u.derivative("x", order=2), u),
            ),
            parameters=(),
            conditions=(),
            regions=(),
        )
        match = "scalar fields"
    with pytest.raises(ValueError, match=match):
        phx.equations.compile_finite_difference_residual(
            problem, grid, equation_targets=targets
        )


@pytest.mark.parametrize(
    "periodic,boundary,cells,count,accuracy,expected",
    [
        (
            True,
            False,
            True,
            64,
            2,
            "78d90504e63807254981b90e208b5e13a902d7bf2ff3145274cad8d2c46a8525",
        ),
        (
            False,
            True,
            False,
            33,
            4,
            "e640dac1bddddc4652bde8f77993cf64a0462d393ae9c899481d384ba700629e",
        ),
    ],
    ids=["periodic-cell-heat", "fourth-order-nodal-dirichlet"],
)
def test_existing_dynamics_compilation_ids(
    periodic: bool, boundary: bool, cells: bool, count: int, accuracy: int, expected: str
) -> None:
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=periodic)
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(phx.equations.PDEField("u", coordinates=("x", "t")),),
        parameters=(phx.equations.PDEParameter("kappa", value=0.2),),
        equations=(
            phx.equations.PDEEquation(
                "heat",
                u.derivative("t"),
                phx.equations.PDEExpression.parameter("kappa") * u.laplacian("x"),
            ),
        ),
        regions=(phx.equations.PDERegion("boundary", "boundary", ("x",)),)
        if boundary
        else (),
        conditions=(
            phx.equations.PDECondition(
                "dirichlet",
                "boundary",
                u,
                target=phx.equations.PDEExpression.constant(0.0),
                region="boundary",
                coordinate="x",
            ),
        )
        if boundary
        else (),
    )
    spec = (
        phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
        if cells
        else phx.discretization.UniformAxisSpec(count)
    )
    grid = phx.discretization.TensorGridPlan((spec,), axis_names=("x",)).prepare(
        jnp.asarray([[0.0], [1.0]], dtype=jnp.float64)
    )
    compiled = phx.equations.compile_finite_difference_pde(
        problem,
        grid,
        policy=phx.equations.FiniteDifferenceCompilationPolicy(accuracy_order=accuracy),
    )
    assert compiled.compilation_id == expected


@pytest.mark.parametrize(
    "expression,degree,dependent",
    [
        (phx.equations.PDEExpression.coordinate_value("x") ** 0.5, 0, False),
        (phx.equations.PDEExpression.parameter("p"), 0, False),
        (phx.equations.PDEExpression.field("u") ** 0, 0, True),
        (phx.equations.PDEExpression.field("u") ** Fraction(3, 1), 3, True),
        (phx.equations.PDEExpression.field("u") ** Fraction(3, 2), None, True),
        (
            phx.equations.PDEExpression.field("u")
            / phx.equations.PDEExpression.field("v"),
            None,
            True,
        ),
        (phx.equations.PDEExpression.field("u").exp(), None, True),
        ((phx.equations.PDEExpression.field("u") ** 3).laplacian("x"), 3, True),
        (
            phx.equations.PDEExpression.constant(2)
            ** phx.equations.PDEExpression.field("u"),
            None,
            True,
        ),
    ],
    ids=[
        "field-free-fractional-power",
        "parameter",
        "zero-power-corrected",
        "exact-integer-power",
        "noninteger-power",
        "field-denominator",
        "nonpolynomial",
        "linear-differential-op",
        "field-dependent-exponent-corrected",
    ],
)
def test_exact_field_analysis(
    expression: phx.equations.PDEExpression, degree: int | None, dependent: bool
) -> None:
    assert field_polynomial_degree(expression) == degree
    assert depends_on_fields(expression) is dependent


def test_spectral_field_free_fractional_coefficient_uses_matrix_free_route() -> None:
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(12),), axis_names=("x",), field_name="u"
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    u = phx.equations.PDEExpression.field("u")
    x = phx.equations.PDEExpression.coordinate_value("x")
    problem = phx.equations.PDEProblemIR(
        coordinates=(
            phx.equations.PDECoordinate("x", "space", periodic=True),
            phx.equations.PDECoordinate("t", "time"),
        ),
        fields=(phx.equations.PDEField("u", coordinates=("x", "t")),),
        equations=(
            phx.equations.PDEEquation(
                "rate", u.derivative("t"), (1.0 + x) ** 0.5 * u + u**0
            ),
        ),
    )
    compiled = phx.equations.compile_spectral_pde(
        problem, space, phx.discretization.PseudospectralMethodPlan()
    )
    nodes = np.asarray(space.axes[0].nodes)
    physical = 0.5 + np.sin(2.0 * np.pi * nodes)
    state = compiled.project_state(jnp.asarray(physical))
    actual = compiled.drift(jnp.zeros((), dtype=jnp.float64), state, {})
    expected = space.project(jnp.asarray(np.sqrt(1.0 + nodes) * physical + 1.0))
    np.testing.assert_allclose(actual, expected, atol=1e-11)


def test_semidiscrete_zero_power_admits_exact_affine_splitting() -> None:
    # Sampling axes exercise the strong-form semidiscrete route.
    axis = phx.discretization.FourierAxisSpec(12).materialize(
        jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(1.0, dtype=jnp.float64)
    )
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        coordinates=(
            phx.equations.PDECoordinate("x", "space", periodic=True),
            phx.equations.PDECoordinate("t", "time"),
        ),
        fields=(phx.equations.PDEField("u", coordinates=("x", "t")),),
        equations=(phx.equations.PDEEquation("rate", u.derivative("t"), u + u**0),),
    )
    compiled = phx.equations.compile_semidiscrete_pde(
        problem, spatial, method="semilinear"
    )
    state = 0.5 + jnp.sin(2.0 * jnp.pi * axis.nodes)
    np.testing.assert_allclose(
        compiled(jnp.zeros((), dtype=jnp.float64), state, {}), state + 1.0, atol=1e-12
    )


@pytest.mark.parametrize(
    "route", ["spectral", "semidiscrete", "fd"], ids=["spectral", "semidiscrete", "fd"]
)
def test_integer_powers_have_finite_exact_jvp_at_zero(route: str) -> None:
    u = phx.equations.PDEExpression.field("u")
    powers = u**0 + u**1 + u**2
    problem = phx.equations.PDEProblemIR(
        coordinates=(
            phx.equations.PDECoordinate("x", "space", periodic=True),
            phx.equations.PDECoordinate("t", "time"),
        ),
        fields=(phx.equations.PDEField("u", coordinates=("x", "t")),),
        equations=(phx.equations.PDEEquation("powers", u.derivative("t"), powers),),
    )
    if route == "spectral":
        problem = replace(
            problem,
            equations=(
                phx.equations.PDEEquation(
                    "powers", u.derivative("t"), u.laplacian("x") + powers
                ),
            ),
        )
        space = phx.discretization.TensorSpectralPlan(
            (phx.discretization.FourierBasisPlan(8),), axis_names=("x",), field_name="u"
        ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
        compiled = phx.equations.compile_spectral_pde(
            problem,
            space,
            phx.discretization.PseudospectralMethodPlan(
                dealiasing=phx.discretization.PaddingDealiasingPlan(2)
            ),
            splitting="direct",
        )
        state = space.project(jnp.zeros((8,), dtype=jnp.float64))
        tangent = space.project(jnp.ones((8,), dtype=jnp.float64))
    elif route == "semidiscrete":
        axis = phx.discretization.FourierAxisSpec(8).materialize(
            jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(1.0, dtype=jnp.float64)
        )
        spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
        compiled = phx.equations.compile_semidiscrete_pde(
            problem, spatial, method="direct"
        )
        state = jnp.zeros((8,), dtype=jnp.float64)
        tangent = jnp.ones((8,), dtype=jnp.float64)
    else:
        problem = replace(
            problem,
            equations=(
                phx.equations.PDEEquation(
                    "powers", u.derivative("t"), u.laplacian("x") + powers
                ),
            ),
        )
        grid = phx.discretization.TensorGridPlan(
            (phx.discretization.UniformCellAxisSpec(8, periodic=True),), axis_names=("x",)
        ).prepare(jnp.asarray([[0.0], [1.0]], dtype=jnp.float64))
        compiled = phx.equations.compile_finite_difference_pde(problem, grid)
        state = compiled.layout.pack({"u": jnp.zeros(grid.shape, dtype=jnp.float64)})
        tangent = compiled.layout.pack({"u": jnp.ones(grid.shape, dtype=jnp.float64)})
    time = jnp.zeros((), dtype=jnp.float64)
    primal, jvp = jax.jvp(lambda value: compiled(time, value, {}), (state,), (tangent,))
    assert np.all(np.isfinite(np.asarray(primal)))
    assert np.all(np.isfinite(np.asarray(jvp)))
    np.testing.assert_allclose(primal, tangent, atol=1e-12)
    np.testing.assert_allclose(jvp, tangent, atol=1e-12)

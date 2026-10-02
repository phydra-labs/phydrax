from dataclasses import dataclass
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._strict import StrictModule
from phydrax._trainable import parameter_field
from phydrax.domain import (
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
    DomainFunction,
)
from phydrax.equations._compile import compile_pde_expression
from phydrax.equations._ir import (
    PDECoordinate,
    PDEEquation,
    PDEExpression,
    PDEField,
    PDEParameter,
    PDEProblemIR,
)
from phydrax.equations._randomized_compile import (
    analyze_randomized_compilation,
    compile_pde_randomized_term,
    RandomizedDifferentialPlan,
)
from phydrax.operators.differential._dimension_estimators import DimensionSamplingPolicy
from phydrax.operators.differential._requests import DerivativeStep
from phydrax.operators.differential._taylor_contracts import (
    TaylorContractionPolicy,
    TaylorContractionResources,
)
from phydrax.typing import PRNGKey


def _domain(size: int = 2, *, label: str = "x") -> Any:
    return phx.domain.HyperRectangle(
        jnp.full((size,), -1.0, dtype=jnp.float64),
        jnp.full((size,), 1.0, dtype=jnp.float64),
        label=label,
    )


def _problem(
    expression: PDEExpression,
    *,
    size: int = 2,
    vector: bool = False,
    parameters: tuple[PDEParameter, ...] = (),
) -> PDEProblemIR:
    return PDEProblemIR(
        coordinates=(PDECoordinate("x", "space", size=size, bounds=(-1.0, 1.0)),),
        fields=(
            PDEField(
                "u",
                representation="vector" if vector else "scalar",
                components=size if vector else 1,
                coordinates=("x",),
            ),
        ),
        parameters=parameters,
        equations=(PDEEquation("equation", expression),),
    )


def _compile(problem: PDEProblemIR, domain: Any, plan: RandomizedDifferentialPlan) -> Any:
    return compile_pde_randomized_term(
        problem,
        "equation",
        plan,
        component=domain.component(),
        sampling=phx.domain.PointSampling(3, layout=phx.domain.SampleLayout((("x",),))),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(17),
    )


def _point_value(
    expression: PDEExpression, function: Any, point: jax.Array, *, vector: bool = False
) -> jax.Array:
    compiled = compile_pde_expression(
        expression,
        _problem(expression, vector=vector),
        fields={"u": function},
        differential_backend="jet",
    )
    if not isinstance(compiled, phx.domain.DomainFunction):
        raise TypeError("A differential expression must compile to a DomainFunction.")
    coordinates = {"x": point}
    return jnp.asarray(
        compiled.func(*(coordinates[label] for label in compiled.deps), key=jr.key(0))
    )


def test_mixed_whole_chain_and_live_parameter_gradient() -> None:
    u = PDEExpression.field("u")
    expression = u.derivative("x", axis=0, order=2).derivative("x", axis=1)
    domain = _domain()
    point = jnp.asarray([0.4, -0.3], dtype=jnp.float64)

    def evaluate(coefficient: jax.Array) -> jax.Array:
        function = domain.Function("x")(lambda x: coefficient * x[0] ** 3 * x[1] ** 2)
        return _point_value(expression, function, point)

    compiled = eqx.filter_jit(jax.value_and_grad(evaluate))
    for coefficient in (0.7, 1.4):
        value, gradient = compiled(jnp.asarray(coefficient, dtype=jnp.float64))
        np.testing.assert_allclose(
            value, coefficient * 12.0 * point[0] * point[1], atol=1e-10
        )
        np.testing.assert_allclose(gradient, 12.0 * point[0] * point[1], atol=1e-10)


def test_g_pinn_linear_residual_and_nested_laplacian() -> None:
    u = PDEExpression.field("u")
    domain = _domain()
    point = jnp.asarray([0.4, -0.3], dtype=jnp.float64)
    function = domain.Function("x")(lambda x: x[0] ** 4 * x[1] ** 2 + x[1] ** 6)
    g_pinn = (u.laplacian("x") + 3.0 * u).derivative("x", axis=0)
    # Coefficients inside an outer derivative are retained in its exact operand.
    expected = (
        24.0 * point[0] * point[1] ** 2
        + 8.0 * point[0] ** 3
        + 12.0 * point[0] ** 3 * point[1] ** 2
    )
    np.testing.assert_allclose(_point_value(g_pinn, function, point), expected, atol=1e-9)
    bilaplacian = u.laplacian("x").laplacian("x")
    np.testing.assert_allclose(
        _point_value(bilaplacian, function, point),
        384.0 * point[1] ** 2 + 48.0 * point[0] ** 2,
        atol=1e-9,
    )

    def evaluate(coefficient: jax.Array) -> jax.Array:
        current = domain.Function("x")(
            lambda x: coefficient * (x[0] ** 4 * x[1] ** 2 + x[1] ** 6)
        )
        return jnp.stack(
            (
                _point_value(g_pinn, current, point),
                _point_value(bilaplacian, current, point),
            )
        )

    gradient = jax.jacrev(evaluate)(jnp.asarray(0.7, dtype=jnp.float64))
    np.testing.assert_allclose(
        gradient,
        jnp.asarray(
            [expected, 384.0 * point[1] ** 2 + 48.0 * point[0] ** 2], dtype=jnp.float64
        ),
        atol=1e-9,
    )


def test_vector_event_component_selection_commutes_with_contraction() -> None:
    u = PDEExpression.field("u")
    domain = _domain()
    function = domain.Function("x")(
        lambda x: jnp.stack((x[0] ** 4 * x[1] ** 2, 2.0 * x[0] ** 3 * x[1] ** 2))
    )
    expression = u.laplacian("x").derivative("x", axis=1).component(1)
    point = jnp.asarray([0.4, -0.3], dtype=jnp.float64)
    np.testing.assert_allclose(
        _point_value(expression, function, point, vector=True),
        24.0 * point[0] * point[1],
        atol=1e-9,
    )


def test_heterogeneous_signed_term_population_exact_and_parameter_gradient() -> None:
    u = PDEExpression.field("u")
    expression = (
        2.0 * u.derivative("x", axis=0, order=3)
        - 0.5 * u.laplacian("x").laplacian("x")
        + 7.0
    )
    problem = _problem(expression)
    domain = _domain()
    # One pure partial plus four compact pair contributions; no shape padding.
    plan = RandomizedDifferentialPlan(
        "dimension",
        backend="jet",
        population="terms",
        dimension_policy=DimensionSamplingPolicy(5, 5),
        loss_mode="u_statistic",
    )
    compiled = _compile(problem, domain, plan)
    batch = compiled.term.sample(key=jr.key(9))

    def loss(coefficient: jax.Array) -> jax.Array:
        function = domain.Function("x")(
            lambda x: coefficient * (x[0] ** 3 + jnp.sum(x**4))
        )
        return compiled.term.loss({"u": function}, batch=batch)

    # Each x^4 contributes 24 to Delta^2; partial^3 additionally depends on x0.
    points = jnp.asarray(batch.collocation["x"].data)

    def target(c: jax.Array) -> jax.Array:
        return jnp.mean((2.0 * c * (6.0 + 24.0 * points[..., 0]) - 24.0 * c + 7.0) ** 2)

    coefficient = jnp.asarray(0.6, dtype=jnp.float64)
    actual = eqx.filter_jit(jax.value_and_grad(loss))(coefficient)
    expected = jax.value_and_grad(target)(coefficient)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-9, atol=1e-9)
    diagnostics = compiled.term.diagnostics(
        {"u": domain.Function("x")(lambda x: coefficient * (x[0] ** 3 + jnp.sum(x**4)))},
        batch=batch,
    )
    assert diagnostics.sampling_design == "exact"
    np.testing.assert_allclose(diagnostics.mean_probe_standard_error, 0.0)


def test_coefficient_inside_derivative_is_differentiated_but_outside_stays_live() -> None:
    u = PDEExpression.field("u")
    a = PDEExpression.parameter("a")
    inside = (a * u).laplacian("x").derivative("x", axis=0)
    outside = a * u.laplacian("x").derivative("x", axis=0)
    problem = _problem(inside, parameters=(PDEParameter("a", functional=True),))
    domain = _domain()
    field = domain.Function("x")(lambda x: x[0] ** 2)
    coefficient_field = domain.Function("x")(lambda x: x[0] ** 3)
    point = jnp.asarray([0.4, 0.2], dtype=jnp.float64)
    compiled = compile_pde_expression(
        inside,
        problem,
        fields={"u": field},
        parameters={"a": coefficient_field},
        differential_backend="jet",
    )
    np.testing.assert_allclose(
        compiled.func(point, key=jr.key(0)), 60.0 * point[0] ** 2, atol=1e-9
    )

    def evaluate(coefficient: jax.Array) -> jax.Array:
        result = compile_pde_expression(
            outside,
            problem,
            fields={"u": domain.Function("x")(lambda x: x[0] ** 3)},
            parameters={"a": coefficient},
            differential_backend="jet",
        )
        return result.func(point, key=jr.key(0))

    np.testing.assert_allclose(
        jax.grad(evaluate)(jnp.asarray(0.3, dtype=jnp.float64)), 6.0, atol=1e-10
    )


def test_compact_coordinate_bilaplacian_population_and_bias_refusals() -> None:
    u = PDEExpression.field("u")
    expression = u.laplacian("x").laplacian("x")
    plan = RandomizedDifferentialPlan(
        "dimension",
        backend="jet",
        dimension_policy=DimensionSamplingPolicy(4, 4),
        loss_mode="u_statistic",
    )
    compiled = _compile(_problem(expression), _domain(), plan)
    field = _domain().Function("x")(lambda x: jnp.sum(x**4))
    samples = compiled.term.diagnostics({"u": field}, key=jr.key(10))
    np.testing.assert_allclose(samples.plug_in_residual_norm, 48.0, atol=1e-9)
    for biased in (expression.exp(), expression * expression, 1.0 / expression):
        report = analyze_randomized_compilation(_problem(biased), "equation", plan)
        assert not report.supported
    vector = u.laplacian("x").dot(u.laplacian("x"))
    vector_report = analyze_randomized_compilation(
        _problem(vector, vector=True),
        "equation",
        RandomizedDifferentialPlan(backend="jet"),
    )
    assert not vector_report.supported
    refused = analyze_randomized_compilation(
        _problem(expression),
        "equation",
        RandomizedDifferentialPlan(
            "dimension",
            backend="jet",
            dimension_policy=DimensionSamplingPolicy(2, 2),
            loss_mode="u_statistic",
        ),
    )
    assert not refused.supported


def test_gaussian_fourth_moment_and_nonnormal_refusal() -> None:
    u = PDEExpression.field("u")
    expression = u.laplacian("x").laplacian("x")
    domain = _domain()
    plan = RandomizedDifferentialPlan(
        "gaussian_bilaplacian",
        backend="jet",
        trace_policy=phx.operators.StochasticTracePolicy(4096, distribution="normal"),
        loss_mode="plug_in",
    )
    compiled = _compile(_problem(expression), domain, plan)
    field = domain.Function("x")(lambda x: jnp.sum(x**4))
    batch = compiled.term.sample(key=jr.key(31))
    samples = compiled.term.residual_evaluator(
        {"u": field}, batch.collocation, batch.left_key
    )
    np.testing.assert_allclose(
        samples.mean, 48.0, atol=6.0 * np.max(np.asarray(samples.standard_error))
    )
    assert samples.sampling_design == "iid"
    with pytest.raises(ValueError):
        RandomizedDifferentialPlan(
            "gaussian_bilaplacian", trace_policy=phx.operators.StochasticTracePolicy(8)
        )


def test_resource_refusal_and_explicit_coordinate_identity() -> None:
    x_domain, y_domain = _domain(label="x"), _domain(label="y")
    domain = phx.domain.ProductDomain(x_domain, y_domain)
    u = PDEExpression.field("u")
    expression = u.derivative("x", axis=0).derivative("y", axis=1)
    problem = PDEProblemIR(
        coordinates=(
            PDECoordinate("x", "space", size=2),
            PDECoordinate("y", "space", size=2),
        ),
        fields=(PDEField("u", coordinates=("x", "y")),),
        equations=(PDEEquation("equation", expression),),
    )
    function = domain.Function("x", "y")(lambda x, y: x[0] ** 2 * y[1] ** 3)
    compiled = compile_pde_expression(
        expression, problem, fields={"u": function}, differential_backend="jet"
    )
    x, y = (
        jnp.asarray([0.4, 0.7], dtype=jnp.float64),
        jnp.asarray([-0.2, 0.3], dtype=jnp.float64),
    )
    np.testing.assert_allclose(
        compiled.func(x, y, key=jr.key(0)), 6.0 * x[0] * y[1] ** 2, atol=1e-10
    )
    refused = analyze_randomized_compilation(
        _problem(u.laplacian("x").laplacian("x")),
        "equation",
        RandomizedDifferentialPlan(
            "gaussian_bilaplacian",
            backend="jet",
            taylor_policy=TaylorContractionPolicy(
                resources=TaylorContractionResources(max_order=3)
            ),
        ),
    )
    assert not refused.supported


def test_exact_singleton_and_nonexact_singleton_uncertainty() -> None:
    domain = _domain(1)
    u = PDEExpression.field("u")
    expression = u.laplacian("x")
    field = domain.Function("x")(lambda x: x[0] ** 2)
    exact = _compile(
        _problem(expression, size=1),
        domain,
        RandomizedDifferentialPlan(
            "dimension",
            backend="jet",
            dimension_policy=DimensionSamplingPolicy(1, 1),
            loss_mode="u_statistic",
        ),
    )
    diagnostics = exact.term.diagnostics({"u": field}, key=jr.key(10))
    np.testing.assert_allclose(diagnostics.objective, 4.0, atol=1e-10)
    np.testing.assert_allclose(diagnostics.mean_probe_standard_error, 0.0)
    assert diagnostics.uncertainty_available
    iid = _compile(
        _problem(expression, size=1),
        domain,
        RandomizedDifferentialPlan(
            "dimension",
            backend="jet",
            dimension_policy=DimensionSamplingPolicy(1, 1, replace=True),
            loss_mode="plug_in",
        ),
    )
    diagnostics = iid.term.diagnostics({"u": field}, key=jr.key(10))
    np.testing.assert_allclose(diagnostics.objective, 4.0, atol=1e-10)
    assert not diagnostics.uncertainty_available
    assert jnp.isnan(diagnostics.mean_probe_standard_error)


def test_distinct_equal_size_coordinates_in_one_population() -> None:
    domain = phx.domain.ProductDomain(_domain(label="x"), _domain(label="y"))
    u = PDEExpression.field("u")
    expression = u.laplacian("x") + 2.0 * u.laplacian("y")
    problem = PDEProblemIR(
        coordinates=(
            PDECoordinate("x", "space", size=2),
            PDECoordinate("y", "space", size=2),
        ),
        fields=(PDEField("u", coordinates=("x", "y")),),
        equations=(PDEEquation("equation", expression),),
    )
    compiled = compile_pde_randomized_term(
        problem,
        "equation",
        RandomizedDifferentialPlan(
            "dimension",
            backend="jet",
            population="terms",
            dimension_policy=DimensionSamplingPolicy(4, 4),
            loss_mode="u_statistic",
        ),
        component=domain.component(),
        sampling=phx.domain.PointSampling(
            2,
            layout=phx.domain.SampleLayout((("x", "y"),)),
        ),
        sampling_mode="fixed",
    )
    field = domain.Function("x", "y")(lambda x, y: jnp.sum(x**2) + 3.0 * jnp.sum(y**2))
    diagnostics = compiled.term.diagnostics({"u": field}, key=jr.key(18))
    np.testing.assert_allclose(diagnostics.objective, 28.0**2, atol=1e-9)


def test_grid_masks_exclude_unselected_collocation_sites() -> None:
    import phydrax.axes as axes

    domain = _domain()
    grid = domain.component().sample(phx.domain.GridSampling({"x": 4}), key=jr.key(0))
    original = grid.coord_mask_by_label["x"]
    mask = jnp.zeros_like(original.data, dtype=jnp.bool_).at[0, :].set(True)
    collocation = phx.domain.GridBatch(
        grid.points,
        dense_structure=grid.dense_structure,
        coord_axes_by_label=grid.coord_axes_by_label,
        coord_mask_by_label={"x": axes.AxisArray(mask, dims=original.dims)},
        coord_geometry_weight_by_label=grid.coord_geometry_weight_by_label,
        coord_geometry_order_by_label=grid.coord_geometry_order_by_label,
        axis_discretization_by_axis=grid.axis_discretization_by_axis,
    )
    u = PDEExpression.field("u")
    expression = u.laplacian("x").derivative("x", axis=0)
    compiled = compile_pde_randomized_term(
        _problem(expression),
        "equation",
        RandomizedDifferentialPlan(
            "dimension",
            backend="jet",
            dimension_policy=DimensionSamplingPolicy(2, 2),
            loss_mode="u_statistic",
        ),
        component=domain.component(),
        sampling=phx.domain.PointSampling(1),
        sampling_mode="fixed",
        fixed_batch=collocation,
    )
    field = domain.Function("x")(lambda x: x[0] ** 4)
    diagnostics = compiled.term.diagnostics({"u": field}, key=jr.key(0))
    coordinate = jnp.asarray(grid.points["x"][0].data)[0]
    np.testing.assert_allclose(diagnostics.objective, (24.0 * coordinate) ** 2, atol=1e-8)
    np.testing.assert_allclose(diagnostics.valid_fraction, 0.25, atol=1e-10)


@final
class _OwnedQuartic(StrictModule):
    coefficient: jax.Array = parameter_field()

    def __call__(
        self, x: jax.Array, *, key: PRNGKey | None = None, iter: object = None
    ) -> jax.Array:
        del key, iter
        # Input AD deliberately does not own this scientific derivative.
        return self.coefficient * jax.lax.stop_gradient(jnp.sum(x**4))

    def derivative_rule_for(self, function: DomainFunction, /) -> DerivativeRule:
        return _QuarticRule(function, self.coefficient)


@final
@dataclass(frozen=True, slots=True)
class _QuarticRule(DerivativeRule):
    source: DomainFunction
    coefficient: jax.Array

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del var, axis, order, mode, backend, basis, periodic
        return None

    def derive_path(
        self,
        steps: tuple[DerivativeStep, ...],
        /,
        *,
        mode: DerivativeMode,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, basis, periodic
        if len(steps) == 2 and all(
            step.kind == "laplacian" and step.variable == "x" for step in steps
        ):
            return DomainFunction(
                domain=self.source.domain, deps=(), func=_QuarticFourth(self.coefficient)
            )
        if (
            len(steps) == 1
            and steps[0].kind == "partial"
            and steps[0].axis == 0
            and steps[0].order == 3
        ):
            return DomainFunction(
                domain=self.source.domain,
                deps=("x",),
                func=_QuarticThird(self.coefficient),
            )
        return None


@final
class _QuarticFourth(StrictModule):
    coefficient: jax.Array = parameter_field()

    def __call__(self, *, key: PRNGKey | None = None, iter: object = None) -> jax.Array:
        del key, iter
        return 48.0 * self.coefficient


@final
class _QuarticThird(StrictModule):
    coefficient: jax.Array = parameter_field()

    def __call__(
        self, x: jax.Array, *, key: PRNGKey | None = None, iter: object = None
    ) -> jax.Array:
        del key, iter
        return 24.0 * self.coefficient * x[0]


def test_native_complete_path_ownership_exact_singleton_and_both_factor_gradients() -> (
    None
):
    domain = _domain()
    u = PDEExpression.field("u")
    expression = u.laplacian("x").laplacian("x")
    compiled = _compile(
        _problem(expression),
        domain,
        RandomizedDifferentialPlan(
            "gaussian_bilaplacian",
            backend="jet",
            prefer_exact=False,
            loss_mode="independent_product",
        ),
    )
    batch = compiled.term.sample(key=jr.key(2))

    def loss(coefficient: jax.Array) -> jax.Array:
        field = domain.Function("x")(_OwnedQuartic(coefficient))
        return compiled.term.loss({"u": field}, batch=batch)

    coefficient = jnp.asarray(0.6, dtype=jnp.float64)
    value, gradient = eqx.filter_jit(jax.value_and_grad(loss))(coefficient)
    np.testing.assert_allclose(value, (48.0 * coefficient) ** 2, atol=1e-9)
    np.testing.assert_allclose(gradient, 2.0 * 48.0**2 * coefficient, atol=1e-9)
    field = domain.Function("x")(_OwnedQuartic(coefficient))
    diagnostics = compiled.term.diagnostics({"u": field}, batch=batch)
    assert diagnostics.num_realizations == 1
    assert diagnostics.sampling_design == "exact"
    np.testing.assert_allclose(
        _point_value(expression, field, jnp.asarray([0.2, 0.3], dtype=jnp.float64)),
        48.0 * coefficient,
        atol=1e-10,
    )


def test_native_explicit_term_can_mix_but_full_sum_only_ownership_refuses() -> None:
    domain = _domain()
    u, v = PDEExpression.field("u"), PDEExpression.field("v")

    def compile_expression(expression: PDEExpression, population_size: int) -> Any:
        problem = PDEProblemIR(
            coordinates=(PDECoordinate("x", "space", size=2),),
            fields=(PDEField("u", coordinates=("x",)), PDEField("v", coordinates=("x",))),
            equations=(PDEEquation("equation", expression),),
        )
        return _compile(
            problem,
            domain,
            RandomizedDifferentialPlan(
                "dimension",
                backend="jet",
                population="terms",
                dimension_policy=DimensionSamplingPolicy(
                    population_size, population_size
                ),
                loss_mode="u_statistic",
            ),
        )

    fields = {
        "u": domain.Function("x")(_OwnedQuartic(jnp.asarray(1.0, dtype=jnp.float64))),
        "v": domain.Function("x")(lambda x: x[0] ** 3),
    }
    explicit = compile_expression(
        u.derivative("x", axis=0, order=3) + v.laplacian("x"), 3
    )
    batch = explicit.term.sample(key=jr.key(4))
    coordinate = jnp.asarray(batch.collocation["x"].data)[..., 0]
    np.testing.assert_allclose(
        explicit.term.loss(fields, batch=batch),
        jnp.mean((30.0 * coordinate) ** 2),
        atol=1e-8,
    )
    implicit = compile_expression(u.laplacian("x").laplacian("x") + v.laplacian("x"), 6)
    with pytest.raises(ValueError):
        implicit.term.loss(fields, key=jr.key(4))


@pytest.mark.parametrize("backend", ["ad", "jet"])
def test_runtime_field_event_shape_cannot_override_scalar_ir(backend: Any) -> None:
    domain = _domain()
    expression = PDEExpression.field("u").laplacian("x")
    compiled = _compile(
        _problem(expression), domain, RandomizedDifferentialPlan(backend=backend)
    )
    wrong_shape = domain.Function("x")(lambda x: jnp.stack((x[0] ** 2, x[1] ** 2)))
    with pytest.raises(ValueError):
        compiled.term.loss({"u": wrong_shape}, key=jr.key(1))


def test_partial_coordinate_population_reports_finite_population_uncertainty() -> None:
    domain = _domain(4)
    expression = PDEExpression.field("u").laplacian("x")
    compiled = _compile(
        _problem(expression, size=4),
        domain,
        RandomizedDifferentialPlan(
            "dimension",
            backend="jet",
            dimension_policy=DimensionSamplingPolicy(4, 2),
            loss_mode="independent_product",
        ),
    )
    field = domain.Function("x")(
        lambda x: jnp.dot(jnp.asarray([1.0, 2.0, 4.0, 8.0], dtype=jnp.float64), x**2)
    )
    batch = compiled.term.sample(key=jr.key(12))
    samples = compiled.term.residual_evaluator(
        {"u": field}, batch.collocation, batch.left_key
    )
    values = np.asarray(samples.values)
    variance = np.sum((values - values.mean(axis=0)) ** 2, axis=0)
    expected = np.sqrt(np.mean((1.0 - 2.0 / 4.0) * variance / 2.0))
    diagnostics = compiled.term.diagnostics({"u": field}, batch=batch)
    assert diagnostics.sampling_design == "finite_population"
    assert diagnostics.population_size == 4
    np.testing.assert_allclose(diagnostics.mean_probe_standard_error, expected, atol=1e-9)


def test_exact_numerical_coefficient_component_has_scalar_zero_derivative() -> None:
    expression = (
        PDEExpression.parameter("a").component(1).derivative("x", axis=0, order=3)
    )
    problem = _problem(
        expression, parameters=(PDEParameter("a", components=2, value=(1.0, 2.0)),)
    )
    coefficient = jnp.asarray([0.5, 0.7], dtype=jnp.float64)

    def evaluate(value: jax.Array) -> jax.Array:
        return compile_pde_expression(
            expression,
            problem,
            fields={},
            parameters={"a": value},
            differential_backend="jet",
        )

    result = evaluate(coefficient)
    assert result.shape == ()
    np.testing.assert_allclose(result, 0.0)
    np.testing.assert_allclose(
        jax.grad(evaluate)(coefficient), jnp.zeros_like(coefficient)
    )


def test_scalar_coordinate_and_vector_coordinate_share_one_exact_curve() -> None:
    space = _domain()
    time = phx.domain.ScalarInterval(-1.0, 1.0, label="t")
    domain = phx.domain.ProductDomain(space, time)
    expression = PDEExpression.field("u").derivative("x", axis=1).derivative("t", order=2)
    problem = PDEProblemIR(
        coordinates=(PDECoordinate("x", "space", size=2), PDECoordinate("t", "time")),
        fields=(PDEField("u", coordinates=("x", "t")),),
        equations=(PDEEquation("equation", expression),),
    )
    field = domain.Function("x", "t")(lambda x, t: x[1] ** 3 * t**4)
    compiled = compile_pde_expression(
        expression, problem, fields={"u": field}, differential_backend="jet"
    )
    x, t = (
        jnp.asarray([0.2, 0.4], dtype=jnp.float64),
        jnp.asarray(-0.3, dtype=jnp.float64),
    )
    np.testing.assert_allclose(
        compiled.func(x, t, key=jr.key(0)), 36.0 * x[1] ** 2 * t**2, atol=1e-10
    )


@final
class _OwnedVectorPolynomial(StrictModule):
    coefficient: jax.Array = parameter_field()
    refused_axes: tuple[int, ...] = eqx.field(static=True, default=())

    def __call__(
        self, x: jax.Array, *, key: PRNGKey | None = None, iter: object = None
    ) -> jax.Array:
        del key, iter
        return self.coefficient * jax.lax.stop_gradient(
            jnp.stack((x[0] ** 3, 2.0 * x[1] ** 3))
        )

    def derivative_rule_for(self, function: DomainFunction, /) -> DerivativeRule:
        return _VectorPolynomialRule(function, self.coefficient, self.refused_axes)


@final
class _VectorPolynomialPartial(StrictModule):
    coefficient: jax.Array = parameter_field()
    orders: tuple[int, int] = eqx.field(static=True)

    def __call__(
        self, x: jax.Array, *, key: PRNGKey | None = None, iter: object = None
    ) -> jax.Array:
        del key, iter

        def cubic(order: int, value: jax.Array) -> jax.Array:
            match order:
                case 0:
                    return value**3
                case 1:
                    return 3.0 * value**2
                case 2:
                    return 6.0 * value
                case 3:
                    return jnp.asarray(6.0, dtype=value.dtype)
                case _:
                    return jnp.asarray(0.0, dtype=value.dtype)

        first = (
            cubic(self.orders[0], x[0])
            if self.orders[1] == 0
            else jnp.asarray(0.0, dtype=x.dtype)
        )
        second = (
            2.0 * cubic(self.orders[1], x[1])
            if self.orders[0] == 0
            else jnp.asarray(0.0, dtype=x.dtype)
        )
        return self.coefficient * jnp.stack((first, second))


@final
@dataclass(frozen=True, slots=True)
class _VectorPolynomialRule(DerivativeRule):
    source: DomainFunction
    coefficient: jax.Array
    refused_axes: tuple[int, ...]

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        if var != "x" or axis is None or axis in self.refused_axes:
            return None
        return self.derive_path(
            (DerivativeStep("partial", var, axis=axis, order=order, backend=backend),),
            mode=mode,
            basis=basis,
            periodic=periodic,
        )

    def derive_path(
        self,
        steps: tuple[DerivativeStep, ...],
        /,
        *,
        mode: DerivativeMode,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, basis, periodic
        if any(
            step.kind != "partial"
            or step.variable != "x"
            or step.axis is None
            or step.axis in self.refused_axes
            for step in steps
        ):
            return None
        orders = (
            sum(step.order for step in steps if step.axis == 0),
            sum(step.order for step in steps if step.axis == 1),
        )
        return DomainFunction(
            domain=self.source.domain,
            deps=("x",),
            func=_VectorPolynomialPartial(self.coefficient, orders),
        )


def test_native_divergence_paths_project_actual_components_and_preserve_order() -> None:
    domain = _domain()
    u = PDEExpression.field("u")
    expression = u.divergence("x").derivative("x", axis=0)
    field = domain.Function("x")(
        _OwnedVectorPolynomial(jnp.asarray(0.7, dtype=jnp.float64))
    )
    point = jnp.asarray([0.3, -0.5], dtype=jnp.float64)
    np.testing.assert_allclose(
        _point_value(expression, field, point, vector=True),
        6.0 * 0.7 * point[0],
        atol=1e-10,
    )
    compiled = _compile(
        _problem(expression, vector=True),
        domain,
        RandomizedDifferentialPlan(backend="jet"),
    )
    diagnostics = compiled.term.diagnostics({"u": field}, key=jr.key(3))
    assert diagnostics.sampling_design == "exact"
    assert diagnostics.num_realizations == 1
    batch = compiled.term.sample(key=jr.key(3))
    coordinate = jnp.asarray(batch.collocation["x"].data)[..., 0]
    np.testing.assert_allclose(
        diagnostics.objective, jnp.mean((6.0 * 0.7 * coordinate) ** 2), atol=1e-9
    )


def test_native_divergence_contributions_mix_with_other_scientific_terms() -> None:
    domain = _domain()
    u, v = PDEExpression.field("u"), PDEExpression.field("v")
    expression = u.divergence("x") + v.derivative("x", axis=0, order=3)
    problem = PDEProblemIR(
        coordinates=(PDECoordinate("x", "space", size=2),),
        fields=(
            PDEField("u", representation="vector", components=2, coordinates=("x",)),
            PDEField("v", coordinates=("x",)),
        ),
        equations=(PDEEquation("equation", expression),),
    )
    compiled = _compile(
        problem,
        domain,
        RandomizedDifferentialPlan(
            "dimension",
            backend="jet",
            population="terms",
            dimension_policy=DimensionSamplingPolicy(3, 3),
            loss_mode="u_statistic",
        ),
    )
    batch = compiled.term.sample(key=jr.key(11))
    points = jnp.asarray(batch.collocation["x"].data)
    exact = 3.0 * points[..., 0] ** 2 + 6.0 * points[..., 1] ** 2

    def loss(coefficient: jax.Array) -> jax.Array:
        return compiled.term.loss(
            {
                "u": domain.Function("x")(_OwnedVectorPolynomial(coefficient)),
                "v": domain.Function("x")(lambda x: x[0] ** 3),
            },
            batch=batch,
        )

    def reference(coefficient: jax.Array) -> jax.Array:
        return jnp.mean((coefficient * exact + 6.0) ** 2)

    coefficient = jnp.asarray(0.6, dtype=jnp.float64)
    actual, expected = (
        eqx.filter_jit(jax.value_and_grad(loss))(coefficient),
        jax.value_and_grad(reference)(coefficient),
    )
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-8)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-8)


def test_partial_native_divergence_ownership_and_case_resources_refuse() -> None:
    domain = _domain()
    expression = PDEExpression.field("u").divergence("x")
    compiled = _compile(
        _problem(expression, vector=True),
        domain,
        RandomizedDifferentialPlan(backend="jet"),
    )
    partial = domain.Function("x")(
        _OwnedVectorPolynomial(jnp.asarray(1.0, dtype=jnp.float64), (1,))
    )
    with pytest.raises(ValueError):
        compiled.term.loss({"u": partial}, key=jr.key(0))
    refused = _compile(
        _problem(expression, vector=True),
        domain,
        RandomizedDifferentialPlan(
            backend="jet",
            taylor_policy=TaylorContractionPolicy(
                resources=TaylorContractionResources(max_linear_terms=1)
            ),
        ),
    )
    field = domain.Function("x")(
        _OwnedVectorPolynomial(jnp.asarray(1.0, dtype=jnp.float64))
    )
    with pytest.raises(ValueError):
        refused.term.loss({"u": field}, key=jr.key(0))


def test_scalar_and_one_component_vector_keep_distinct_operator_types() -> None:
    domain = _domain(1)
    expression = PDEExpression.field("u").divergence("x")
    scalar = domain.Function("x")(lambda x: x[0] ** 3)
    with pytest.raises(ValueError):
        compile_pde_expression(
            expression,
            _problem(expression, size=1),
            fields={"u": scalar},
            differential_backend="jet",
        )
    compiled = _compile(
        _problem(expression, size=1, vector=True),
        domain,
        RandomizedDifferentialPlan(
            "dimension",
            backend="jet",
            dimension_policy=DimensionSamplingPolicy(1, 1),
            loss_mode="u_statistic",
        ),
    )
    batch = compiled.term.sample(key=jr.key(0))
    vector = domain.Function("x")(lambda x: jnp.stack((x[0] ** 3,)))
    coordinate = jnp.asarray(batch.collocation["x"].data)[..., 0]
    np.testing.assert_allclose(
        compiled.term.loss({"u": vector}, batch=batch),
        jnp.mean(9.0 * coordinate**4),
        atol=1e-9,
    )


@pytest.mark.parametrize("backend", ["ad", "jet"])
def test_one_component_vector_hutchinson_on_scalar_coordinate_has_live_gradients(
    backend: Any,
) -> None:
    domain = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    expression = PDEExpression.field("u").divergence("x")
    compiled = _compile(
        _problem(expression, size=1, vector=True),
        domain,
        RandomizedDifferentialPlan(
            backend=backend,
            trace_policy=phx.operators.StochasticTracePolicy(8),
        ),
    )
    batch = compiled.term.sample(key=jr.key(7))
    coordinate = jnp.asarray(batch.collocation["x"].data)
    fourth_moment = jnp.mean(coordinate**4)

    def loss(coefficient: jax.Array) -> jax.Array:
        field = domain.Function("x")(lambda x: jnp.stack((coefficient * x**3,)))
        return compiled.term.loss({"u": field}, batch=batch)

    evaluate = eqx.filter_jit(jax.value_and_grad(loss))
    for scalar in (0.4, 0.8):
        coefficient = jnp.asarray(scalar, dtype=jnp.float64)
        value, gradient = evaluate(coefficient)
        np.testing.assert_allclose(value, 9.0 * coefficient**2 * fourth_moment, atol=1e-9)
        np.testing.assert_allclose(
            gradient, 18.0 * coefficient * fourth_moment, atol=1e-9
        )

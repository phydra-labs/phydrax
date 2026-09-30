"""Consumer contracts for scientific form typing and smooth PDE lowering."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.domain import DomainFunction, HyperRectangle
from phydrax.equations import (
    compile_pde_expression,
    infer_expression_type,
    pde_ir_from_dict,
    pde_ir_to_dict,
    PDECoordinate,
    PDEEquation,
    PDEExpression,
    PDEField,
    PDEFormGeometry,
    PDEFormTrace,
    PDEProblemIR,
    PDERegion,
    validate_pde_ir,
)
from phydrax.exterior import FormType, FormValueSpec
from phydrax.metrix import CoordinateChart, euclidean_metric
from phydrax.operators.differential import DomainDifferentialForm


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


def _problem() -> PDEProblemIR:
    return PDEProblemIR(
        coordinates=(PDECoordinate("x", "space", size=3),),
        fields=(
            PDEField(
                "u",
                coordinates=("x",),
                form=FormValueSpec(FormType(3, 0), proxy="scalar"),
            ),
        ),
    )


def test_form_degree_and_twist_typing() -> None:
    problem = _problem()
    u = PDEExpression.field("u")
    first = infer_expression_type(u.exterior_derivative(), problem)
    second = infer_expression_type(u.exterior_derivative().exterior_derivative(), problem)
    dual = infer_expression_type(u.hodge_star(), problem)
    if first.form is None or second.form is None or dual.form is None:
        raise AssertionError("Exterior typing must preserve scientific form metadata.")
    assert (first.form.form_type.degree, first.form.proxy, first.representation) == (
        1,
        "circulation",
        "vector",
    )
    assert (second.form.form_type.degree, second.form.proxy, second.representation) == (
        2,
        "flux",
        "pseudovector",
    )
    assert (
        dual.form.form_type.degree,
        dual.form.form_type.twist,
        dual.representation,
    ) == (3, "twisted", "scalar")
    equation = PDEEquation("closed", u.exterior_derivative().exterior_derivative())
    validate_pde_ir(
        PDEProblemIR(problem.coordinates, problem.fields, equations=(equation,))
    )


def test_form_proxy_crossing_is_refused() -> None:
    problem = PDEProblemIR(
        coordinates=(PDECoordinate("x", "space", size=3),),
        fields=(
            PDEField(
                "a",
                representation="tensor",
                components=3,
                coordinates=("x",),
                form=FormValueSpec(FormType(3, 1), proxy="components"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="proxy"):
        infer_expression_type(PDEExpression.field("a").curl("x"), problem)
    with pytest.raises(ValueError, match="match its form proxy"):
        PDEField(
            "bad",
            representation="vector",
            components=3,
            form=FormValueSpec(FormType(3, 2), proxy="flux"),
        )


def test_form_serialization_roundtrip() -> None:
    original = _problem()
    u = PDEExpression.field("u")
    problem = PDEProblemIR(
        original.coordinates,
        original.fields,
        equations=(PDEEquation("closed", u.exterior_derivative().exterior_derivative()),),
    )
    payload = pde_ir_to_dict(problem)
    restored = pde_ir_from_dict(payload)
    assert pde_ir_to_dict(restored) == payload
    assert restored.canonical_hash == problem.canonical_hash
    field_form = restored.fields[0].form
    if field_form is None:
        raise AssertionError("A restored PDE field lost its scientific form metadata.")
    original_form = problem.fields[0].form
    if original_form is None:
        raise AssertionError("The fixture must declare a scientific form.")
    assert field_form.value_spec_id == original_form.value_spec_id
    payload["fields"][0].pop("form")
    with pytest.raises(ValueError, match="canonical fields"):
        pde_ir_from_dict(payload)


def test_smooth_exterior_lowering_matches_polynomial() -> None:
    domain = HyperRectangle(
        jnp.full((3,), -2.0, dtype=jnp.float64),
        jnp.full((3,), 2.0, dtype=jnp.float64),
        label="x",
    )
    chart = CoordinateChart("polynomial", ("x", "y", "z"))

    def coefficients(x: Array, /, *, key: Array | None = None) -> Array:
        del key
        return jnp.reshape(x[0] ** 2 + x[1] * x[2], (1,))

    carrier = DomainDifferentialForm(
        DomainFunction(domain=domain, deps=("x",), func=coefficients),
        chart=chart,
        degree=0,
        var="x",
    )
    problem = _problem()
    expression = PDEExpression.field("u").exterior_derivative()
    compiled = compile_pde_expression(expression, problem, fields={"u": carrier})
    if not isinstance(compiled, DomainFunction):
        raise AssertionError("Smooth PDE lowering must produce a DomainFunction.")
    point = jnp.asarray([0.4, -0.2, 0.7], dtype=jnp.float64)
    np.testing.assert_allclose(
        eqx.filter_jit(compiled.func)(point), [0.8, 0.7, -0.2], rtol=1e-12, atol=1e-12
    )
    laplacian = compile_pde_expression(
        PDEExpression.field("u").laplacian("x"),
        problem,
        fields={"u": carrier},
        form_geometry=PDEFormGeometry(metric=euclidean_metric(chart)),
    )
    if not isinstance(laplacian, DomainFunction):
        raise AssertionError("Smooth Laplacian lowering must produce a DomainFunction.")
    np.testing.assert_allclose(
        eqx.filter_jit(laplacian.func)(point), 2.0, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    "operation", ["top_derivative", "zero_codifferential", "dimension"]
)
def test_form_degree_and_dimension_refusals(operation: str) -> None:
    if operation == "dimension":
        problem = PDEProblemIR(
            (PDECoordinate("x", "space", size=2),),
            (
                PDEField(
                    "u",
                    coordinates=("x",),
                    form=FormValueSpec(FormType(3, 0), proxy="scalar"),
                ),
            ),
        )
        with pytest.raises(ValueError, match="spatial coordinate dimension"):
            validate_pde_ir(problem)
        return
    problem = _problem()
    u = PDEExpression.field("u")
    expression = (
        u.codifferential()
        if operation == "zero_codifferential"
        else u.exterior_derivative()
        .exterior_derivative()
        .exterior_derivative()
        .exterior_derivative()
    )
    with pytest.raises(ValueError):
        infer_expression_type(expression, problem)


def _linear_form_bindings() -> tuple[
    PDEProblemIR, dict[str, DomainFunction | DomainDifferentialForm]
]:
    domain = HyperRectangle(
        jnp.full((3,), -2.0, dtype=jnp.float64),
        jnp.full((3,), 2.0, dtype=jnp.float64),
        label="x",
    )
    chart = CoordinateChart("cartan", ("x", "y", "z"))

    def coefficients(x: Array, /, *, key: Array | None = None) -> Array:
        del key
        return jnp.stack((x[1], x[2], x[0]))

    def vector(x: Array, /, *, key: Array | None = None) -> Array:
        del x, key
        return jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)

    carrier = DomainDifferentialForm(
        DomainFunction(domain=domain, deps=("x",), func=coefficients),
        chart=chart,
        degree=1,
        var="x",
    )
    tangent = DomainFunction(domain=domain, deps=("x",), func=vector)
    problem = PDEProblemIR(
        (PDECoordinate("x", "space", size=3),),
        (
            PDEField(
                "a",
                representation="tensor",
                components=3,
                coordinates=("x",),
                form=FormValueSpec(FormType(3, 1), proxy="components"),
            ),
            PDEField("v", representation="vector", components=3, coordinates=("x",)),
        ),
        regions=(PDERegion("wall", "boundary", ("x",)),),
    )
    return problem, {"a": carrier, "v": tangent}


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        (PDEExpression.field("a").interior_product(PDEExpression.field("v")), (2.4,)),
        (
            PDEExpression.field("a").lie_derivative(PDEExpression.field("v")),
            (2.0, 3.0, 1.0),
        ),
        (
            PDEExpression.field("a").wedge(
                PDEExpression.field("a").exterior_derivative()
            ),
            (-0.9,),
        ),
    ),
    ids=("contraction", "cartan-lie", "wedge"),
)
def test_smooth_algebra_lowering_has_analytic_values(
    expression: PDEExpression, expected: tuple[float, ...]
) -> None:
    problem, bindings = _linear_form_bindings()
    point = jnp.asarray([0.4, -0.2, 0.7], dtype=jnp.float64)
    compiled = compile_pde_expression(expression, problem, fields=bindings)
    if not isinstance(compiled, DomainFunction):
        raise AssertionError("Smooth form algebra must lower to a DomainFunction.")
    np.testing.assert_allclose(
        eqx.filter_jit(compiled.func)(point), expected, rtol=1e-12, atol=1e-12
    )


def test_smooth_trace_lowering_has_analytic_value() -> None:
    problem, bindings = _linear_form_bindings()
    a = PDEExpression.field("a")
    point = jnp.asarray([0.4, -0.2], dtype=jnp.float64)

    wall = HyperRectangle(
        jnp.full((2,), -2.0, dtype=jnp.float64),
        jnp.full((2,), 2.0, dtype=jnp.float64),
        label="s",
    )

    def embedding(s: Array, /, *, key: Array | None = None) -> Array:
        del key
        return jnp.stack((s[0], s[1], jnp.asarray(0.0, dtype=s.dtype)))

    mapping = DomainFunction(domain=wall, deps=("s",), func=embedding)
    geometry = PDEFormGeometry(
        traces={
            "wall": PDEFormTrace(
                mapping,
                CoordinateChart("wall", ("x", "y")),
                source_var="s",
                coorientation=1,
            )
        }
    )
    trace = compile_pde_expression(
        a.trace("wall"), problem, fields=bindings, form_geometry=geometry
    )
    if not isinstance(trace, DomainFunction):
        raise AssertionError("A smooth trace must lower to a boundary DomainFunction.")
    np.testing.assert_allclose(
        eqx.filter_jit(trace.func)(point), [-0.2, 0.0], rtol=1e-12, atol=1e-12
    )


def test_form_proxy_and_arithmetic_preserve_declared_derivatives() -> None:
    from phydrax.domain._derivative import (
        CallbackDerivativeRule,
        DerivativeBackend,
        DerivativeBasis,
        DerivativeMode,
    )
    from phydrax.operators.differential import grad

    domain = HyperRectangle(
        jnp.full((3,), -2.0, dtype=jnp.float64),
        jnp.full((3,), 2.0, dtype=jnp.float64),
        label="x",
    )
    chart = CoordinateChart("external-polynomial", ("x", "y", "z"))

    def coefficients(x: Array, /, *, key: Array | None = None) -> Array:
        del key
        return jax.lax.stop_gradient(jnp.reshape(x[0] ** 2, (1,)))

    def derive(
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, backend, basis, periodic
        if var != "x" or order != 1:
            return None

        def derivative(x: Array, /, *, key: Array | None = None) -> Array:
            del key
            values = jnp.stack(
                (
                    2.0 * x[0],
                    jnp.asarray(0.0, dtype=x.dtype),
                    jnp.asarray(0.0, dtype=x.dtype),
                )
            )
            return values[None, :] if axis is None else jnp.reshape(values[axis], (1,))

        return DomainFunction(domain=domain, deps=("x",), func=derivative)

    field = DomainDifferentialForm(
        DomainFunction(
            domain=domain,
            deps=("x",),
            func=coefficients,
            derivative_rule=CallbackDerivativeRule(derive),
        ),
        chart=chart,
        degree=0,
        var="x",
    )
    problem = _problem()
    u = PDEExpression.field("u")
    proxy = compile_pde_expression(u, problem, fields={"u": field})
    if not isinstance(proxy, DomainFunction):
        raise AssertionError("A typed field must lower to its physical proxy.")
    point = jnp.asarray([0.4, -0.2, 0.7], dtype=jnp.float64)
    np.testing.assert_allclose(
        grad(proxy, var="x", backend="ad").func(point),
        [0.8, 0.0, 0.0],
        rtol=1e-12,
        atol=1e-12,
    )
    summed = compile_pde_expression(
        (u + u).exterior_derivative(), problem, fields={"u": field}
    )
    if not isinstance(summed, DomainFunction):
        raise AssertionError("A differentiated form sum must lower to a proxy.")
    np.testing.assert_allclose(
        summed.func(point), [1.6, 0.0, 0.0], rtol=1e-12, atol=1e-12
    )

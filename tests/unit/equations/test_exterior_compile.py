"""Consumer-visible exterior lowering, with analytic and coordinate oracles."""

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import TensorGridPlan, UniformCellAxisSpec
from phydrax.discretization._cell_complex import simplicial_cell_complex
from phydrax.discretization._cochain import CochainDiscretization
from phydrax.discretization._cochain_hodge import DiagonalHodge, SparseHodge
from phydrax.discretization._cubical_whitney import CubicalSplineWhitneyKernel
from phydrax.discretization._structured_cochain import StructuredCochainBridge
from phydrax.equations._exterior_compile import (
    compile_exterior_pde,
    CompiledExteriorPDE,
    ExteriorPDERealization,
)
from phydrax.equations._ir import (
    PDECoordinate,
    PDEEquation,
    PDEExpression,
    PDEField,
    PDEProblemIR,
    PDERegion,
)
from phydrax.exterior._complex import DiscreteForm
from phydrax.exterior._form_type import FormType, FormValueSpec
from phydrax.exterior._products import WhitneyProductPlan
from phydrax.exterior._traces import trace_map


_D0 = np.asarray([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]], dtype=np.float64)
_D1 = np.asarray([[1.0, -1.0, 1.0]], dtype=np.float64)
_M0 = np.asarray([2.0, 3.0, 5.0], dtype=np.float64)
_M1 = np.asarray([7.0, 11.0, 13.0], dtype=np.float64)
_M2 = np.asarray([17.0], dtype=np.float64)


def _triangle() -> CochainDiscretization:
    topology = simplicial_cell_complex(
        (
            np.arange(3, dtype=np.int32)[:, None],
            np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
    )
    return CochainDiscretization(
        topology,
        (DiagonalHodge(_M0), DiagonalHodge(_M1), DiagonalHodge(_M2)),
        numeric_revision="exterior-compile-test-metric",
    )


def _field(
    name: str, degree: int, /, *, dimension: int = 2, fiber: tuple[int, ...] = ()
) -> PDEField:
    spec = FormValueSpec(
        FormType(dimension, degree, fiber_shape=fiber), proxy="components"
    )
    return PDEField(
        name,
        representation="tensor",
        components=prod(spec.value_shape),
        coordinates=("x", "y"),
        form=spec,
    )


def _problem(
    fields: tuple[PDEField, ...],
    equations: tuple[PDEEquation, ...],
    /,
    *,
    regions: tuple[PDERegion, ...] = (),
) -> PDEProblemIR:
    return PDEProblemIR(
        (PDECoordinate("x", "space"), PDECoordinate("y", "space")),
        fields,
        equations=equations,
        regions=regions,
    )


def test_exterior_lowering_matches_native_operators() -> None:
    realization = _triangle()
    u, e, f = (PDEExpression.field(name) for name in ("u", "e", "f"))
    problem = _problem(
        (_field("u", 0), _field("e", 1), _field("f", 2)),
        (
            PDEEquation("d0", u.exterior_derivative()),
            PDEEquation("d1", e.exterior_derivative()),
            PDEEquation("delta1", e.codifferential()),
            PDEEquation("delta2", f.codifferential()),
            PDEEquation("star1", e.hodge_star()),
            PDEEquation("star_square", e.hodge_star().hodge_star()),
            PDEEquation("dual_d", e.hodge_star().exterior_derivative()),
            PDEEquation("dual_delta", e.hodge_star().codifferential()),
            PDEEquation("nilpotency", u.exterior_derivative().exterior_derivative()),
            PDEEquation("laplacian", e.laplacian("x")),
        ),
    )
    values = {
        "u": jnp.asarray([1.0, -2.0, 4.0]),
        "e": jnp.asarray([3.0 + 1.0j, -1.0 + 2.0j, 2.0 - 4.0j], dtype=jnp.complex128),
        "f": jnp.asarray([5.0]),
    }
    compiled = compile_exterior_pde(
        problem, realization, fields=values, boundary="absolute"
    )
    result = compiled.residuals()
    delta1 = (_D0.T * _M1) / _M0[:, None]
    delta2 = (_D1.T * _M2) / _M1[:, None]
    expected = {
        "d0": _D0 @ np.asarray(values["u"]),
        "d1": _D1 @ np.asarray(values["e"]),
        "delta1": delta1 @ np.asarray(values["e"]),
        "delta2": delta2 @ np.asarray(values["f"]),
        "star1": _M1 * np.asarray(values["e"]),
        "star_square": -np.asarray(values["e"]),
        "dual_d": -_D0.T @ (_M1 * np.asarray(values["e"])),
        "dual_delta": _M2 * (_D1 @ np.asarray(values["e"])),
        "nilpotency": np.zeros((1,), dtype=np.float64),
        "laplacian": -(_D0 @ delta1 + delta2 @ _D1) @ np.asarray(values["e"]),
    }
    for name, oracle in expected.items():
        np.testing.assert_allclose(result[name], oracle, rtol=1e-12, atol=1e-12)


def test_exterior_prepared_dynamic_reuse() -> None:
    realization = _triangle()
    u = PDEExpression.field("u")
    problem = _problem(
        (_field("u", 0),),
        (PDEEquation("screened", u.exterior_derivative().codifferential() + u),),
    )
    initial = jnp.asarray([1.0, 2.0, -3.0])
    compiled = compile_exterior_pde(
        problem, realization, fields={"u": initial}, boundary="absolute"
    )

    @eqx.filter_jit
    def evaluate(plan: CompiledExteriorPDE, value: Array) -> Array:
        return plan.residuals({"u": value})["screened"]

    oracle = ((_D0.T * _M1) @ _D0) / _M0[:, None] + np.eye(3, dtype=np.float64)
    changed = jnp.asarray([-4.0, 0.5, 7.0])
    np.testing.assert_allclose(
        evaluate(compiled, initial), oracle @ np.asarray(initial), atol=1e-12
    )
    np.testing.assert_allclose(
        evaluate(compiled, changed), oracle @ np.asarray(changed), atol=1e-12
    )
    rebound = eqx.tree_at(lambda plan: plan.fields, compiled, (changed,))
    np.testing.assert_allclose(
        rebound.residuals()["screened"], oracle @ np.asarray(changed), atol=1e-12
    )
    assert rebound.compilation_id == compiled.compilation_id

    def loss(value: Array) -> Array:
        result = evaluate(compiled, value)
        return jnp.vdot(result, result).real

    np.testing.assert_allclose(
        jax.grad(loss)(changed),
        2 * oracle.T @ oracle @ np.asarray(changed),
        rtol=1e-12,
        atol=1e-11,
    )


def test_exterior_relative_boundary_matches_restricted_complex() -> None:
    base = _triangle()
    metric = np.asarray(
        [[2.0, 0.5, 0.7], [0.5, 3.0, 1.0], [0.7, 1.0, 4.0]], dtype=np.float64
    )
    rows, columns = np.triu_indices(3)
    hodge = SparseHodge(rows, columns, metric[rows, columns], 3)
    realization = CochainDiscretization(
        base.topology,
        (hodge, DiagonalHodge(_M1), DiagonalHodge(_M2)),
        boundary_masks=(
            np.asarray([True, False, False]),
            np.zeros(3, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
        ),
        numeric_revision="exterior-relative-test-metric",
    )
    e = PDEExpression.field("e")
    problem = _problem((_field("e", 1),), (PDEEquation("delta", e.codifferential()),))
    value = jnp.asarray([1.0, 2.0, -1.0])
    plan = compile_exterior_pde(
        problem, realization, fields={"e": value}, boundary="relative"
    )
    rhs = _D0[:, 1:].T @ (_M1 * np.asarray(value))
    expected = np.concatenate(
        (np.zeros((1,), dtype=np.float64), np.linalg.solve(metric[1:, 1:], rhs))
    )
    result = plan.residuals()["delta"]
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
    full_inverse_then_mask = np.linalg.solve(metric, _D0.T @ (_M1 * np.asarray(value)))
    assert not np.allclose(np.asarray(result)[1:], full_inverse_then_mask[1:])
    assert result[0] == 0.0


def _product_binding() -> tuple[StructuredCochainBridge, ExteriorPDERealization]:
    grid = TensorGridPlan(
        (UniformCellAxisSpec(2), UniformCellAxisSpec(2)), axis_names=("x", "y")
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    bridge = StructuredCochainBridge(grid)
    plan = WhitneyProductPlan(
        bridge, CubicalSplineWhitneyKernel(bridge), quadrature_order=3
    )
    return bridge, ExteriorPDERealization(bridge, products=plan)


def test_exterior_products_interior_and_lie_have_analytic_chain_values() -> None:
    bridge, binding = _product_binding()
    a, b, c, vector = (PDEExpression.field(name) for name in ("a", "b", "c", "X"))
    problem = _problem(
        (
            _field("a", 1),
            _field("b", 1),
            _field("c", 1),
            PDEField("X", representation="vector", components=2, coordinates=("x", "y")),
        ),
        (
            PDEEquation("wedge", a.wedge(b)),
            PDEEquation("interior", a.interior_product(vector)),
            PDEEquation("lie", c.lie_derivative(vector)),
        ),
    )
    horizontal, vertical = bridge.orientation_shapes[1]
    x = bridge.grid.structured_axes[0].point_coordinates
    values = {
        "a": bridge.pack(1, (jnp.full(horizontal, 0.5), jnp.zeros(vertical))),
        "b": bridge.pack(1, (jnp.zeros(horizontal), jnp.full(vertical, 0.5))),
        "c": bridge.pack(
            1, (jnp.zeros(horizontal), jnp.broadcast_to(0.5 * x[:, None], vertical))
        ),
        "X": jnp.asarray([2.0, -1.0]),
    }
    compiled = compile_exterior_pde(problem, binding, fields=values, boundary="absolute")
    result = compiled.residuals()
    np.testing.assert_allclose(
        result["wedge"], np.full(bridge.cell_counts[2], 0.25), atol=1e-12
    )
    np.testing.assert_allclose(
        result["interior"], np.full(bridge.cell_counts[0], 2.0), atol=1e-12
    )
    expected_lie = bridge.pack(1, (jnp.zeros(horizontal), jnp.ones(vertical)))
    np.testing.assert_allclose(result["lie"], expected_lie, atol=1e-12)


def test_exterior_matrix_wedge_preserves_noncommutative_fibers() -> None:
    bridge, binding = _product_binding()
    a, b = PDEExpression.field("a"), PDEExpression.field("b")
    problem = _problem(
        (_field("a", 1, fiber=(2, 2)), _field("b", 1, fiber=(2, 2))),
        (
            PDEEquation("ab", a.wedge(b, product="matrix")),
            PDEEquation("ba", b.wedge(a, product="matrix")),
        ),
    )
    horizontal, vertical = bridge.orientation_shapes[1]
    dx = bridge.pack(1, (jnp.full(horizontal, 0.5), jnp.zeros(vertical)))
    dy = bridge.pack(1, (jnp.zeros(horizontal), jnp.full(vertical, 0.5)))
    left = jnp.asarray([[0.0, 1.0], [0.0, 0.0]])
    right = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    plan = compile_exterior_pde(
        problem,
        binding,
        fields={"a": dx[:, None, None] * left, "b": dy[:, None, None] * right},
        boundary="absolute",
    )
    result = plan.residuals()
    expected_ab = jnp.broadcast_to(0.25 * (left @ right), (bridge.cell_counts[2], 2, 2))
    expected_ba = jnp.broadcast_to(-0.25 * (right @ left), (bridge.cell_counts[2], 2, 2))
    np.testing.assert_allclose(result["ab"], expected_ab, atol=1e-12)
    np.testing.assert_allclose(result["ba"], expected_ba, atol=1e-12)


def test_exterior_trace_lowers_boundary_derivative_and_stokes() -> None:
    base = _triangle()
    realization = CochainDiscretization(
        base.topology,
        (DiagonalHodge(_M0), DiagonalHodge(_M1), DiagonalHodge(_M2)),
        boundary_masks=(
            np.ones(3, dtype=np.bool_),
            np.ones(3, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
        ),
        numeric_revision="exterior-trace-test-metric",
    )
    binding = ExteriorPDERealization(realization, traces={"wall": trace_map(realization)})
    u = PDEExpression.field("u")
    traced_d = u.exterior_derivative().trace("wall")
    boundary_d = u.trace("wall").exterior_derivative()
    problem = _problem(
        (_field("u", 0),),
        (
            PDEEquation("boundary_d", boundary_d),
            PDEEquation("commuting", boundary_d, traced_d),
        ),
        regions=(PDERegion("wall", "boundary", ("x", "y")),),
    )
    plan = compile_exterior_pde(
        problem, binding, fields={"u": jnp.asarray([2.0, 5.0, -1.0])}, boundary="absolute"
    )
    result = plan.residuals()
    np.testing.assert_allclose(
        result["boundary_d"], np.asarray([3.0, 3.0, -6.0]), atol=1e-12
    )
    np.testing.assert_allclose(
        result["commuting"], np.zeros(3, dtype=np.float64), atol=1e-12
    )
    np.testing.assert_allclose(jnp.sum(result["boundary_d"]), 0.0, atol=1e-12)


def test_exterior_binding_refuses_equal_shape_wrong_identity_and_missing_geometry() -> (
    None
):
    realization = _triangle()
    u = PDEExpression.field("u")
    problem = _problem((_field("u", 0),), (PDEEquation("d", u.exterior_derivative()),))
    wrong = DiscreteForm("other-realization", FormType(2, 0), jnp.zeros(3))
    with pytest.raises(ValueError, match="identity"):
        compile_exterior_pde(
            problem, realization, fields={"u": wrong}, boundary="absolute"
        )
    product_problem = _problem((_field("u", 0),), (PDEEquation("product", u.wedge(u)),))
    with pytest.raises(ValueError, match="WhitneyProductPlan"):
        compile_exterior_pde(
            product_problem, realization, fields={"u": jnp.zeros(3)}, boundary="absolute"
        )

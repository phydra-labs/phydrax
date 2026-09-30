#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from itertools import combinations
from math import factorial
from typing import final

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

from phydrax.discretization import (
    CochainDiscretization,
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.discretization._cell_complex import simplicial_cell_complex
from phydrax.discretization._cochain_hodge import DiagonalHodge
from phydrax.discretization._cochain_orientation import reorient_cell_complex
from phydrax.domain import HyperRectangle
from phydrax.exterior._complex import AbstractDeRhamComplex, ComplexBoundary
from phydrax.exterior._de_rham import (
    CellParameterization,
    DeRhamBridge,
    integrate_form,
    metric_dual_hodges,
    simplicial_parameterizations,
    structured_parameterizations,
    validate_de_rham_commutation,
)
from phydrax.exterior._form_type import FormTwist
from phydrax.linalg import HilbertComplex
from phydrax.metrix import CoordinateChart, diagonal_metric, DifferentialForm
from phydrax.operators.differential import DomainDifferentialForm


@final
class _TwistedRealization(AbstractDeRhamComplex):
    """The same incidence coordinates with a twisted primal orientation line."""

    cochain: CochainDiscretization

    def __init__(self, cochain: CochainDiscretization, /) -> None:
        self.cochain = cochain

    @property
    def dimension(self) -> int:
        return self.cochain.dimension

    @property
    def primal_twist(self) -> FormTwist:
        return "twisted"

    @property
    def realization_id(self) -> str:
        return f"twisted:{self.cochain.realization_id}"

    def hilbert_complex(
        self, /, *, boundary: ComplexBoundary = "absolute"
    ) -> HilbertComplex:
        return self.cochain.hilbert_complex(boundary=boundary)

    def hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        return self.cochain.hodge_star(degree, values)

    def inverse_hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        return self.cochain.inverse_hodge_star(degree, values)


def _simplex_bridge(dimension: int, *, order: int = 5) -> DeRhamBridge:
    rows = tuple(
        np.asarray(tuple(combinations(range(dimension + 1), degree + 1)), dtype=np.int32)
        for degree in range(dimension + 1)
    )
    topology = simplicial_cell_complex(rows)
    coordinates = jnp.concatenate((jnp.zeros((1, dimension)), jnp.eye(dimension)))
    complex_ = CochainDiscretization(
        topology,
        tuple(
            DiagonalHodge(jnp.ones((entities.count,)))
            for entities in topology.entity_sets
        ),
    )
    chart = CoordinateChart("simplex", tuple(f"x{index}" for index in range(dimension)))
    return DeRhamBridge(
        complex_, chart, simplicial_parameterizations(topology, coordinates, order=order)
    )


@pytest.mark.parametrize("dimension", (1, 2, 3, 4))
def test_polynomial_top_form_matches_exact_simplex_moment(dimension: int) -> None:
    bridge = _simplex_bridge(dimension)
    form = DifferentialForm(
        lambda point: jnp.asarray([point[0] ** 2]), chart=bridge.chart, degree=dimension
    )
    integrated = integrate_form(form, bridge)
    # Dirichlet moment: integral x_0^2 over the standard n-simplex = 2/(n+2)!.
    np.testing.assert_allclose(
        integrated.values, [2 / factorial(dimension + 2)], atol=1e-13, rtol=1e-13
    )
    compiled = jax.jit(lambda: integrate_form(form, bridge).values)()
    np.testing.assert_allclose(compiled, integrated.values, atol=1e-13)


@pytest.mark.parametrize("dimension", (1, 2, 3, 4))
def test_polynomial_exterior_derivative_commutes_on_every_simplex_degree(
    dimension: int,
) -> None:
    bridge = _simplex_bridge(dimension)
    for degree in range(dimension):
        component_count = len(tuple(combinations(range(dimension), degree)))
        form = DifferentialForm(
            lambda point: (
                jnp.arange(1, component_count + 1, dtype=jnp.float64) * jnp.sum(point**3)
            ),
            chart=bridge.chart,
            degree=degree,
        )
        evidence = validate_de_rham_commutation(form, bridge, tolerance=2e-12)
        assert bool(evidence.valid)
        assert float(evidence.maximum_residual) < 2e-12


def test_cell_reorientation_negates_only_the_reoriented_integrals() -> None:
    bridge = _simplex_bridge(2)
    if not isinstance(bridge.complex, CochainDiscretization):
        raise RuntimeError(
            "The simplex fixture must supply its canonical cell realization."
        )
    signs = (np.ones(3), np.asarray([-1.0, 1.0, -1.0]), np.asarray([-1.0]))
    topology = reorient_cell_complex(bridge.complex.topology, signs)
    coordinates = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    complex_ = CochainDiscretization(
        topology,
        tuple(
            DiagonalHodge(jnp.ones((entities.count,)))
            for entities in topology.entity_sets
        ),
    )
    reoriented = DeRhamBridge(
        complex_,
        bridge.chart,
        simplicial_parameterizations(topology, coordinates, order=5),
    )
    form = DifferentialForm(
        lambda point: jnp.asarray([point[0], point[0] ** 2]), chart=bridge.chart, degree=1
    )
    np.testing.assert_allclose(
        integrate_form(form, reoriented).values,
        signs[1] * integrate_form(form, bridge).values,
        atol=1e-13,
    )
    assert bool(validate_de_rham_commutation(form, reoriented, tolerance=1e-12).valid)


def test_domain_form_integration_and_derivative_have_exact_polynomial_values() -> None:
    bridge = _simplex_bridge(2)
    domain = HyperRectangle(jnp.asarray([-1.0, -1.0]), jnp.asarray([2.0, 2.0]), label="x")

    @domain.Function("x")
    def coefficients(point: Array) -> Array:
        return jnp.asarray([0.0, point[0] ** 2])

    form = DomainDifferentialForm(coefficients, chart=bridge.chart, degree=1, var="x")
    np.testing.assert_allclose(
        integrate_form(form, bridge).values, [0.0, 0.0, 1 / 3], atol=1e-13
    )
    assert bool(validate_de_rham_commutation(form, bridge, tolerance=1e-12).valid)


def test_structured_integration_uses_tensor_cell_order_and_signed_axes() -> None:
    grid = TensorGridPlan(
        (UniformCellAxisSpec(2, periodic=False), UniformCellAxisSpec(3, periodic=False)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0, 3.0]]))
    structured = StructuredCochainBridge(grid)
    chart = CoordinateChart("box", ("x", "y"))
    bridge = DeRhamBridge(
        structured.cochain, chart, structured_parameterizations(structured, order=3)
    )
    form = DifferentialForm(
        lambda point: jnp.asarray([point[0], -2 * point[1]]), chart=chart, degree=1
    )
    horizontal, vertical = structured.unpack(1, integrate_form(form, bridge).values)
    np.testing.assert_allclose(
        horizontal,
        np.broadcast_to(np.asarray([[0.5], [1.5]]), horizontal.shape),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        vertical,
        np.broadcast_to(np.asarray([[-1.0, -3.0, -5.0]]), vertical.shape),
        atol=1e-13,
    )
    assert bool(validate_de_rham_commutation(form, bridge, tolerance=1e-12).valid)


def test_metric_dual_hodges_integrate_both_primal_and_dual_metric_measures() -> None:
    bridge = _simplex_bridge(1)
    dual_vertices = CellParameterization(
        1,
        2,
        1,
        lambda cell, reference: jnp.asarray([0.5 * (cell + reference[0])]),
        lambda cell, reference: jnp.asarray([[0.5]]),
        jnp.asarray([[0.5]]),
        jnp.ones(1),
        jnp.ones(2),
    )
    dual_edge = CellParameterization(
        0,
        1,
        1,
        lambda cell, reference: jnp.asarray([0.5]),
        lambda cell, reference: jnp.zeros((1, 0)),
        jnp.zeros((1, 0)),
        jnp.ones(1),
        jnp.ones(1),
    )
    metric = diagonal_metric(lambda point: jnp.asarray([4.0]), chart=bridge.chart)
    hodges = metric_dual_hodges(bridge, metric, dual=(dual_vertices, dual_edge))
    np.testing.assert_allclose(hodges[0].weights, [1.0, 1.0], atol=1e-13)
    np.testing.assert_allclose(hodges[1].weights, [0.5], atol=1e-13)

    def edge_hodge(scale: Array) -> Array:
        varying_metric = diagonal_metric(
            lambda point: jnp.asarray([scale**2]), chart=bridge.chart
        )
        return metric_dual_hodges(
            bridge, varying_metric, dual=(dual_vertices, dual_edge)
        )[1].weights[0]

    value, derivative = jax.jit(jax.value_and_grad(edge_hodge))(jnp.asarray(2.0))
    np.testing.assert_allclose(value, 0.5, atol=1e-13)
    np.testing.assert_allclose(derivative, -0.25, atol=1e-13)


def test_twist_mismatch_cannot_reinterpret_primal_integrals_as_dual_cells() -> None:
    bridge = _simplex_bridge(2)
    form = DifferentialForm(
        lambda point: jnp.asarray([1.0]), chart=bridge.chart, degree=2, twist="twisted"
    )
    with pytest.raises(ValueError, match="primal twist"):
        integrate_form(form, bridge)


def test_twisted_density_reflection_uses_absolute_jacobian() -> None:
    source = _simplex_bridge(1)
    if not isinstance(source.complex, CochainDiscretization):
        raise RuntimeError("The fixture requires a cell realization.")
    reflected = CellParameterization(
        1,
        1,
        1,
        lambda cell, reference: 1 - reference,
        lambda cell, reference: -jnp.ones((1, 1)),
        jnp.asarray([[0.5]]),
        jnp.ones(1),
        jnp.ones(1),
    )
    bridge = DeRhamBridge(
        _TwistedRealization(source.complex),
        source.chart,
        (source.parameterizations[0], reflected),
    )
    form = DifferentialForm(
        lambda point: jnp.asarray([2.0]), chart=source.chart, degree=1, twist="twisted"
    )
    np.testing.assert_allclose(integrate_form(form, bridge).values, [2.0], atol=1e-13)


def test_embedded_twisted_integration_requires_and_respects_coorientation() -> None:
    source = _simplex_bridge(1)
    if not isinstance(source.complex, CochainDiscretization):
        raise RuntimeError("The fixture requires a cell realization.")
    chart = CoordinateChart("embedded", ("x", "y"))
    vertices = CellParameterization(
        0,
        2,
        2,
        lambda cell, reference: jnp.asarray([cell, 0.0]),
        lambda cell, reference: jnp.zeros((2, 0)),
        jnp.zeros((1, 0)),
        jnp.ones(1),
        jnp.ones(2),
        coorientation_signs=jnp.ones(2),
    )
    form = DifferentialForm(
        lambda point: jnp.asarray([3.0, 0.0]), chart=chart, degree=1, twist="twisted"
    )
    for coorientation in (None, 1, -1):
        edge = CellParameterization(
            1,
            1,
            2,
            lambda cell, reference: jnp.asarray([reference[0], 0.0]),
            lambda cell, reference: jnp.asarray([[1.0], [0.0]]),
            jnp.asarray([[0.5]]),
            jnp.ones(1),
            jnp.ones(1),
            coorientation_signs=None
            if coorientation is None
            else jnp.asarray([coorientation]),
        )
        bridge = DeRhamBridge(
            _TwistedRealization(source.complex), chart, (vertices, edge)
        )
        if coorientation is None:
            with pytest.raises(ValueError, match="coorientation"):
                integrate_form(form, bridge)
        else:
            values = jax.jit(lambda: integrate_form(form, bridge).values)()
            np.testing.assert_allclose(values, [3 * coorientation], atol=1e-13)


def test_zero_cell_quadrature_cannot_scale_or_reverse_point_values() -> None:
    with pytest.raises(ValueError, match="Zero-cell sampling"):
        CellParameterization(
            0,
            1,
            1,
            lambda cell, reference: jnp.asarray([0.0]),
            lambda cell, reference: jnp.zeros((1, 0)),
            jnp.zeros((1, 0)),
            jnp.asarray([2.0]),
            jnp.ones(1),
        )
    with pytest.raises(ValueError, match="Zero-cell sampling"):
        CellParameterization(
            0,
            1,
            1,
            lambda cell, reference: jnp.asarray([0.0]),
            lambda cell, reference: jnp.zeros((1, 0)),
            jnp.zeros((1, 0)),
            jnp.ones(1),
            -jnp.ones(1),
        )

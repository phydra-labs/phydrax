#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._trainable import partition_trainable
from phydrax.nn.models import (
    BSplineEdgeBasis,
    BSplineGrid,
    BSplineGridBank,
    KAN,
    OrthogonalPolynomialEdgeBasis,
    RationalBSplineEdgeBasis,
    RationalBSplineEdgeParameters,
    TrainableBSplineGrid,
)


@pytest.mark.parametrize("degree", (2, 3, 4))
def test_bspline_kan_identity_and_boundary_jacobian(degree):
    basis = BSplineEdgeBasis(degree=degree, num_intervals=6)
    model = KAN(
        in_size="scalar",
        out_size="scalar",
        hidden_sizes=(),
        edge_basis=basis,
        scale_mode="none",
        init="identity",
        skip_connection=False,
        use_bias=False,
        key=jr.key(0),
    )
    assert model.layers[0].bias is None
    assert model.layers[0].scales is None
    points = jnp.linspace(-1.0, 1.0, 17)

    values = jax.jit(lambda values_: jax.vmap(model)(values_))(points)
    derivatives = jax.vmap(jax.grad(model))(points)

    assert np.allclose(np.asarray(values), np.asarray(points), rtol=1e-11, atol=1e-11)
    assert np.allclose(np.asarray(derivatives), 1.0, rtol=1e-10, atol=1e-10)


def test_per_input_bspline_grids_preserve_identity_and_locality():
    grids = (
        BSplineGrid.open_uniform(3, 4),
        BSplineGrid(
            jnp.asarray([-1.0, -1.0, -1.0, -1.0, -0.86, -0.31, 0.57, 1.0, 1.0, 1.0, 1.0]),
            3,
        ),
    )
    basis = BSplineEdgeBasis(grid=BSplineGridBank.from_grids(grids))
    model = KAN(
        in_size=2,
        out_size=2,
        hidden_sizes=(),
        edge_basis=basis,
        scale_mode="none",
        init="identity",
        skip_connection=False,
        use_bias=False,
        key=jr.key(20),
    )
    inputs = jnp.asarray([0.23, -0.41])
    jacobian = jax.jacrev(model)(inputs)
    coefficient_gradient = jax.grad(
        lambda coefficients: jnp.sum(
            basis.evaluate(coefficients, inputs[None, :].repeat(2, 0))
        )
    )(model.layers[0].coeffs)

    assert np.allclose(np.asarray(model(inputs)), np.asarray(inputs), atol=2e-12)
    assert np.allclose(np.asarray(jacobian), np.eye(2), atol=2e-11)
    assert np.array_equal(
        np.asarray(jnp.count_nonzero(jnp.abs(coefficient_gradient) > 1e-12, axis=-1)),
        np.full((2, 2), 4),
    )


def test_per_input_grid_specification_preserves_scan_execution():
    basis = BSplineEdgeBasis(degree=3, num_intervals=5, per_input=True)
    key = jr.key(21)
    loop = KAN(
        in_size=3,
        out_size=2,
        width_size=5,
        depth=4,
        edge_basis=basis,
        scan=False,
        key=key,
    )
    scanned = KAN(
        in_size=3,
        out_size=2,
        width_size=5,
        depth=4,
        edge_basis=basis,
        scan=True,
        key=key,
    )
    inputs = jnp.asarray([0.12, -0.34, 0.56])

    assert scanned._scan_enabled
    assert all(
        layer.edge_basis.grid.num_grids == layer.in_size for layer in scanned.layers
    )
    assert np.allclose(
        np.asarray(eqx.filter_jit(scanned)(inputs)),
        np.asarray(eqx.filter_jit(loop)(inputs)),
        atol=2e-12,
    )


def test_trainable_bspline_grid_participates_in_kan_gradients_and_scan():
    basis = BSplineEdgeBasis(
        grid=TrainableBSplineGrid.open_uniform(3, 6),
        knot_entropy_weight=0.05,
        knot_neighbor_weight=0.02,
    )
    model = KAN(
        in_size=3,
        out_size=2,
        width_size=5,
        depth=4,
        edge_basis=basis,
        scan=True,
        key=jr.key(22),
    )
    inputs = jnp.asarray([0.13, -0.27, 0.41])
    _, gradient = eqx.filter_value_and_grad(
        lambda candidate: (
            jnp.sum(candidate(inputs) ** 2) + 1e-3 * candidate.regularization_loss()
        )
    )(model)
    trainable, fixed = partition_trainable(model)

    assert model._scan_enabled
    assert trainable.layers[0].edge_basis.grid.raw_span_logits is not None
    assert fixed.layers[0].edge_basis.grid.raw_span_logits is None
    assert all(
        np.all(np.isfinite(np.asarray(layer.edge_basis.grid.raw_span_logits)))
        for layer in gradient.layers
    )
    assert np.isfinite(np.asarray(eqx.filter_jit(model)(inputs))).all()


def test_trainable_grid_logits_optimize_without_losing_order():
    initial = TrainableBSplineGrid.open_uniform(3, 6, minimum_span=0.01)
    target = eqx.tree_at(
        lambda grid: grid.raw_span_logits,
        initial,
        jnp.asarray([-1.4, -0.6, 0.8, 1.2, 0.3, -0.3]),
    )
    coefficients = jr.normal(jr.key(23), (1, 1, initial.coefficient_count))
    query = jnp.linspace(-0.95, 0.95, 96)

    def evaluate(grid):
        edge_basis = BSplineEdgeBasis(grid=grid)
        return jax.vmap(
            lambda value: edge_basis.evaluate(coefficients, jnp.asarray([[value]]))[0, 0]
        )(query)

    target_values = evaluate(target)

    def loss(logits):
        grid = eqx.tree_at(lambda value: value.raw_span_logits, initial, logits)
        return jnp.mean((evaluate(grid) - target_values) ** 2)

    step = jax.jit(lambda logits: logits - 0.4 * jax.grad(loss)(logits))
    logits = initial.raw_span_logits
    initial_loss = float(loss(logits))
    for _ in range(60):
        logits = step(logits)
    optimized = eqx.tree_at(
        lambda grid: grid.raw_span_logits,
        initial,
        logits,
    )

    assert float(loss(logits)) < 0.2 * initial_loss
    assert np.all(np.diff(np.asarray(optimized.breakpoints)) >= optimized.minimum_span)


def test_rational_bspline_identity_scan_and_parameter_gradients():
    basis = RationalBSplineEdgeBasis(degree=3, num_intervals=4)
    model = KAN(
        in_size=2,
        out_size=2,
        width_size=3,
        depth=3,
        edge_basis=basis,
        init="identity",
        scale_mode="none",
        skip_connection=False,
        use_bias=False,
        scan=True,
        key=jr.key(24),
    )
    inputs = jnp.asarray([0.21, -0.37])
    value = model(inputs)
    jacobian = jax.jacrev(model)(inputs)
    _, gradient = eqx.filter_value_and_grad(
        lambda candidate: jnp.sum(candidate(inputs) ** 2)
    )(model)

    assert model._scan_enabled
    assert np.allclose(np.asarray(value), np.asarray(inputs), atol=2e-12)
    assert np.allclose(np.asarray(jacobian), np.eye(2), atol=2e-11)
    assert isinstance(model.layers[0].coeffs, RationalBSplineEdgeParameters)
    assert np.all(np.isfinite(np.asarray(gradient.layers[0].coeffs.control_values)))
    assert np.all(np.isfinite(np.asarray(gradient.layers[0].coeffs.raw_log_weights)))


def test_rational_bspline_wins_equal_parameter_reciprocal_fit():
    nodes = jnp.linspace(-1.0, 1.0, 512)
    evaluation = jnp.linspace(-1.0, 1.0, 4096)
    target = lambda values: 1.0 / (1.0 + 0.98 * values)
    rational_grid = BSplineGrid.open_uniform(3, 3)
    fit_plan = phx.operators.BSplineInterpolationPlan(
        degree=3,
        mode="least_squares",
    )
    numerator = phx.operators.fit_bspline(
        nodes,
        jnp.ones(nodes.shape),
        plan=fit_plan,
        grid=rational_grid,
    ).coefficients
    denominator = phx.operators.fit_bspline(
        nodes,
        1.0 + 0.98 * nodes,
        plan=fit_plan,
        grid=rational_grid,
    ).coefficients
    centered_log_weights = jnp.log(denominator) - jnp.mean(jnp.log(denominator))
    raw_log_weights = jnp.arctanh(centered_log_weights / 4.0)
    parameters = RationalBSplineEdgeParameters(
        (numerator / denominator)[None, None, :],
        raw_log_weights[None, None, :],
    )
    rational_basis = RationalBSplineEdgeBasis(grid=rational_grid)
    rational_values = jax.vmap(
        lambda value: rational_basis.evaluate(parameters, jnp.asarray([[value]]))[0, 0]
    )(evaluation)
    polynomial = phx.operators.fit_bspline(
        nodes,
        target(nodes),
        plan=phx.operators.BSplineInterpolationPlan(
            degree=3,
            num_intervals=8,
            mode="least_squares",
        ),
    )
    expected = target(evaluation)
    rational_error = jnp.linalg.norm(rational_values - expected) / jnp.linalg.norm(
        expected
    )
    polynomial_error = jnp.linalg.norm(
        polynomial(evaluation) - expected
    ) / jnp.linalg.norm(expected)

    assert parameters.control_values.size + parameters.raw_log_weights.size - 1 == 11
    assert polynomial.coefficients.size == 11
    assert float(rational_error) < 1e-11
    assert float(rational_error) < 1e-8 * float(polynomial_error)


def test_rational_regularizer_reduces_to_polynomial_energy_at_unit_weights():
    polynomial = BSplineEdgeBasis(degree=3, num_intervals=4)
    rational = RationalBSplineEdgeBasis(
        degree=3,
        num_intervals=4,
        weight_magnitude_weight=0.0,
        weight_variation_weight=0.0,
        denominator_weight=0.0,
    )
    controls = jr.normal(jr.key(25), (2, 3, polynomial.coefficient_count))
    parameters = RationalBSplineEdgeParameters(
        controls,
        jnp.zeros(controls.shape),
    )

    assert float(rational.regularization(parameters)) == pytest.approx(
        float(polynomial.regularization(controls)),
        rel=2e-11,
        abs=2e-11,
    )


@pytest.mark.parametrize("regularization_order", (1, 2, 3))
def test_rational_grid_bank_preserves_every_regularization_order_through_degree(
    regularization_order,
):
    grids = (
        BSplineGrid.open_uniform(3, 4),
        BSplineGrid(
            jnp.asarray([-1.0, -1.0, -1.0, -1.0, -0.72, -0.08, 0.61, 1.0, 1.0, 1.0, 1.0]),
            3,
        ),
    )
    bank = BSplineGridBank.from_grids(grids)
    polynomial = BSplineEdgeBasis(
        grid=bank,
        regularization_order=regularization_order,
    )
    rational = RationalBSplineEdgeBasis(
        grid=bank,
        regularization_order=regularization_order,
        weight_magnitude_weight=0.0,
        weight_variation_weight=0.0,
        denominator_weight=0.0,
    )
    controls = jr.normal(jr.key(26), (1, 2, bank.coefficient_count))
    parameters = RationalBSplineEdgeParameters(
        controls,
        jnp.zeros(controls.shape),
    )

    assert float(rational.regularization(parameters)) == pytest.approx(
        float(polynomial.regularization(controls)),
        rel=2e-11,
        abs=2e-11,
    )


@pytest.mark.parametrize(
    "family",
    ("chebyshev", "legendre", "hermite", "hermite_e", "laguerre"),
)
def test_orthogonal_polynomial_families_have_exact_affine_initialization(family):
    basis = OrthogonalPolynomialEdgeBasis(degree=1, family=family)
    identity = basis.initialize_coefficients(1, 1, "identity", jr.key(30))
    default = basis.initialize_coefficients(1, 1, "default", jr.key(31))

    def evaluate(coefficients, point):
        return basis.evaluate(coefficients, jnp.asarray([[point]]))[0, 0]

    points = jnp.linspace(-1.0, 1.0, 9)
    identity_values = jax.jit(jax.vmap(lambda point: evaluate(identity, point)))(points)
    default_values = jax.vmap(lambda point: evaluate(default, point))(points)
    default_slopes = jax.vmap(jax.grad(lambda point: evaluate(default, point)))(points)

    assert jnp.allclose(identity_values, points, rtol=1e-12, atol=1e-12)
    assert evaluate(default, jnp.asarray(0.0)) == pytest.approx(0.0, abs=1e-14)
    assert jnp.allclose(default_values, default_slopes[0] * points, atol=1e-12)
    assert jnp.allclose(default_slopes, default_slopes[0], atol=1e-12)
    assert basis.regularization(identity) == pytest.approx(0.0, abs=1e-20)


def test_orthogonal_kan_clipping_preserves_endpoint_derivatives():
    model = KAN(
        in_size="scalar",
        out_size="scalar",
        hidden_sizes=(),
        edge_basis=OrthogonalPolynomialEdgeBasis(degree=1),
        scale_mode="none",
        init="identity",
        skip_connection=False,
        use_bias=False,
        key=jr.key(1),
    )

    assert float(jax.grad(model)(jnp.asarray(-1.0))) == pytest.approx(1.0)
    assert float(jax.grad(model)(jnp.asarray(1.0))) == pytest.approx(1.0)
    assert float(jax.grad(model)(jnp.asarray(-1.1))) == pytest.approx(0.0)
    assert float(jax.grad(model)(jnp.asarray(1.1))) == pytest.approx(0.0)


def test_bspline_edge_coefficient_gradients_are_span_local():
    basis = BSplineEdgeBasis(degree=3, num_intervals=8)
    coefficients = jnp.zeros((2, 3, basis.coefficient_count))
    inputs = jnp.full((2, 3), 0.13)

    gradient = jax.grad(lambda values: jnp.sum(basis.evaluate(values, inputs)))(
        coefficients
    )
    active_counts = jnp.count_nonzero(jnp.abs(gradient) > 1e-12, axis=-1)

    assert np.array_equal(np.asarray(active_counts), np.full((2, 3), 4))


def test_bspline_grid_is_excluded_from_trainable_partition():
    model = KAN(
        in_size=2,
        out_size=3,
        hidden_sizes=(),
        edge_basis=BSplineEdgeBasis(degree=3, num_intervals=5),
        key=jr.key(2),
    )

    trainable, fixed = partition_trainable(model)

    assert trainable.layers[0].edge_basis.grid is None
    assert fixed.layers[0].edge_basis.grid is not None
    assert np.array_equal(
        np.asarray(fixed.layers[0].edge_basis.grid.knots),
        np.asarray(model.layers[0].edge_basis.grid.knots),
    )


def test_bspline_regularization_is_sobolev_energy():
    basis = BSplineEdgeBasis(
        degree=3,
        num_intervals=6,
        regularization_order=2,
    )
    affine = basis.initialize_coefficients(1, 1, "identity", jr.key(3))
    curved = affine.at[..., basis.coefficient_count // 2].add(1.0)

    assert float(basis.regularization(affine)) == pytest.approx(0.0, abs=1e-20)
    assert float(basis.regularization(curved)) > 0.0


def test_bspline_kan_scan_matches_loop_and_jacobian_is_finite():
    basis = BSplineEdgeBasis(degree=3, num_intervals=6)
    key = jr.key(4)
    loop = KAN(
        in_size=3,
        out_size=2,
        width_size=5,
        depth=4,
        edge_basis=basis,
        scan=False,
        key=key,
    )
    scanned = KAN(
        in_size=3,
        out_size=2,
        width_size=5,
        depth=4,
        edge_basis=basis,
        scan=True,
        key=key,
    )
    inputs = jnp.asarray([0.1, -0.2, 0.3])

    loop_value = eqx.filter_jit(loop)(inputs)
    scanned_value = eqx.filter_jit(scanned)(inputs)
    jacobian = jax.jacrev(scanned)(inputs)

    assert scanned._scan_enabled
    assert np.allclose(np.asarray(scanned_value), np.asarray(loop_value))
    assert jacobian.shape == (2, 3)
    assert np.all(np.isfinite(np.asarray(jacobian)))


def test_kan_rejects_mismatched_basis_schedule():
    with pytest.raises(ValueError, match="edge_basis must have 3 entries"):
        KAN(
            in_size=2,
            out_size=1,
            width_size=4,
            depth=2,
            edge_basis=(BSplineEdgeBasis(),),
            key=jr.key(5),
        )

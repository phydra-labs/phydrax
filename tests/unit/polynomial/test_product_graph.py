import itertools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._polynomial import ProductGraphPlan, project_tuple_coefficients


pytestmark = pytest.mark.strict_jax

# Requested monomials as variable-index products with their exponent vectors over
# four variables; the exponents are the independent oracle.
_REQUESTS = ((), (0, 0, 0), (2, 1, 1), (1, 0), (3,), (0, 1, 3))
_EXPONENTS = {
    (): (0, 0, 0, 0),
    (0, 0, 0): (3, 0, 0, 0),
    (2, 1, 1): (0, 2, 1, 0),
    (1, 0): (1, 1, 0, 0),
    (3,): (0, 0, 0, 1),
    (0, 1, 3): (1, 1, 0, 1),
}


def _direct_products(
    points: np.ndarray, requests: tuple[tuple[int, ...], ...]
) -> np.ndarray:
    exponents = np.asarray([_EXPONENTS[request] for request in requests], dtype=np.int64)
    return np.prod(points[..., None, :] ** exponents, axis=-1)


@pytest.mark.parametrize(
    "lane_shape", ((), (7,), (2, 3)), ids=("scalar", "vector", "matrix")
)
def test_term_values_match_direct_products(lane_shape: tuple[int, ...]) -> None:
    plan = ProductGraphPlan(4, _REQUESTS, maximum_nodes=32)
    points = np.random.default_rng(3).normal(size=(*lane_shape, 4))

    values = np.asarray(plan.evaluate(jnp.asarray(points)))
    requested = values[..., [plan.term_index(request) for request in _REQUESTS]]

    assert values.shape == (*lane_shape, len(plan.terms))
    np.testing.assert_allclose(
        requested, _direct_products(points, _REQUESTS), rtol=1e-14, atol=1e-15
    )
    np.testing.assert_array_equal(values[..., plan.term_index(())], np.ones(lane_shape))


def _polynomial_coefficients(plan: ProductGraphPlan) -> np.ndarray:
    # p(x) = x0^3 + 2 x0 x1 + 3 x1^2 x2 + 5
    coefficients = np.zeros((len(plan.terms),), dtype=np.float64)
    coefficients[plan.term_index((0, 0, 0))] = 1.0
    coefficients[plan.term_index((1, 0))] = 2.0
    coefficients[plan.term_index((1, 2, 1))] = 3.0
    coefficients[plan.term_index(())] = 5.0
    return coefficients


def _polynomial_gradient(x: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            3.0 * x[0] ** 2 + 2.0 * x[1],
            2.0 * x[0] + 6.0 * x[1] * x[2],
            3.0 * x[1] ** 2,
            0.0,
        ]
    )


def _polynomial_hessian(x: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [6.0 * x[0], 2.0, 0.0, 0.0],
            [2.0, 6.0 * x[2], 6.0 * x[1], 0.0],
            [0.0, 6.0 * x[1], 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )


@pytest.mark.parametrize(
    "point",
    (np.zeros(4), np.asarray([0.7, -1.3, 0.4, 2.1])),
    ids=("zero", "generic"),
)
def test_derivatives_match_closed_form_including_zero_inputs(point: np.ndarray) -> None:
    plan = ProductGraphPlan(4, _REQUESTS, maximum_nodes=32)
    coefficients = jnp.asarray(_polynomial_coefficients(plan))
    direction = jnp.asarray([0.3, -0.8, 1.7, 0.5])

    def polynomial(x: jax.Array) -> jax.Array:
        return plan.contract(x, coefficients)

    x = jnp.asarray(point)
    gradient = jax.grad(polynomial)(x)
    _, hvp = jax.jvp(jax.grad(polynomial), (x,), (direction,))

    assert np.all(np.isfinite(np.asarray(hvp)))
    np.testing.assert_allclose(
        gradient, _polynomial_gradient(point), rtol=1e-14, atol=0.0
    )
    np.testing.assert_allclose(
        hvp, _polynomial_hessian(point) @ np.asarray(direction), rtol=1e-14, atol=1e-15
    )


def test_bilinear_term_has_exact_hessian_at_origin() -> None:
    plan = ProductGraphPlan(2, ((0, 1), (0, 0, 0)), maximum_nodes=8)

    def bilinear(x: jax.Array) -> jax.Array:
        return plan.evaluate(x)[plan.term_index((1, 0))]

    def cubic(x: jax.Array) -> jax.Array:
        return plan.evaluate(x)[plan.term_index((0, 0, 0))]

    origin = jnp.zeros((2,))
    np.testing.assert_array_equal(jax.hessian(bilinear)(origin), [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(jax.grad(cubic)(origin), [0.0, 0.0])
    np.testing.assert_array_equal(jax.hessian(cubic)(origin), np.zeros((2, 2)))


def test_requests_merge_with_deterministic_multiplicity_and_identity() -> None:
    requests = ((1, 0), (2,), (0, 1), (2, 0, 1), (2,), (1, 0, 2), (1, 0))
    plan = ProductGraphPlan(3, requests, maximum_nodes=16)
    reordered = ProductGraphPlan(
        3, ((0, 2, 1), (2,), (0, 1)), maximum_nodes=16, maximum_degree=3
    )

    assert plan.terms == ((2,), (0, 1), (0, 1, 2))
    assert plan.request_terms == (1, 0, 1, 2, 0, 2, 1)
    assert plan.term_requests == (2, 3, 2)
    assert reordered.terms == plan.terms
    assert reordered.term_requests == (1, 1, 1)
    assert reordered.plan_id == plan.plan_id
    assert ProductGraphPlan(4, requests, maximum_nodes=16).plan_id != plan.plan_id


def test_shared_prefixes_bound_node_count_and_live_bytes() -> None:
    requests = ((0, 1), (0, 1, 2), (3, 1, 0))
    # Nodes: 1, x0, x0 x1, x0 x1 x2, x0 x1 x3.
    plan = ProductGraphPlan(4, requests, maximum_nodes=5)

    assert plan.node_count == 5
    assert plan.maximum_degree == 3
    assert plan.live_bytes(10, 8) == 10 * 5 * 8
    with pytest.raises(ValueError, match="requires 5 nodes"):
        ProductGraphPlan(4, requests, maximum_nodes=4)


def test_total_degree_plan_enumerates_every_monomial_in_canonical_order() -> None:
    plan = ProductGraphPlan.total_degree(3, 2, include_constant=True, maximum_nodes=10)
    upper = ProductGraphPlan.total_degree(3, 3, minimum_degree=2, maximum_nodes=20)

    assert plan.terms == (
        (),
        (0,),
        (1,),
        (2,),
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    )
    assert plan.node_count == math.comb(5, 2)
    assert len(upper.terms) == math.comb(4, 2) + math.comb(5, 3)
    assert {len(term) for term in upper.terms} == {2, 3}
    assert upper.node_count == math.comb(6, 3)


def test_coefficient_updates_reuse_one_compiled_plan() -> None:
    plan = ProductGraphPlan.total_degree(3, 3, include_constant=True, maximum_nodes=64)
    generator = np.random.default_rng(11)
    points = generator.normal(size=(5, 3))
    first = generator.normal(size=(len(plan.terms), 2))
    second = generator.normal(size=(len(plan.terms), 2))
    exponents = np.asarray(
        [[term.count(variable) for variable in range(3)] for term in plan.terms],
        dtype=np.int64,
    )
    basis = np.prod(points[:, None, :] ** exponents, axis=-1)

    @jax.jit
    def project(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jnp.asarray(points)
        return plan.contract(x, a), plan.contract(x, b)

    observed_first, observed_second = project(jnp.asarray(first), jnp.asarray(second))
    np.testing.assert_allclose(observed_first, basis @ first, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(observed_second, basis @ second, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("lane_tile", (1, 4, 32), ids=("unit", "ragged", "oversized"))
def test_lane_tiling_matches_untiled_evaluation(lane_tile: int) -> None:
    plan = ProductGraphPlan(4, _REQUESTS, maximum_nodes=32)
    generator = np.random.default_rng(5)
    points = jnp.asarray(generator.normal(size=(11, 2, 4)))
    coefficients = jnp.asarray(generator.normal(size=(len(plan.terms), 3)))

    np.testing.assert_allclose(
        plan.evaluate(points, lane_tile=lane_tile),
        plan.evaluate(points),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        plan.contract(points, coefficients, lane_tile=lane_tile),
        plan.contract(points, coefficients),
        rtol=1e-13,
        atol=1e-13,
    )


def test_total_degree_refuses_combinatorial_size_before_enumeration() -> None:
    with pytest.raises(ValueError, match=f"requires {math.comb(208, 8)} nodes"):
        ProductGraphPlan.total_degree(200, 8, maximum_nodes=10**4)


def _brute_force_projection(
    tensor: np.ndarray, variable_count: int, degree: int
) -> dict[tuple[int, ...], np.ndarray]:
    sums: dict[tuple[int, ...], np.ndarray] = {}
    touched: set[tuple[int, ...]] = set()
    for index in itertools.product(range(variable_count), repeat=degree):
        term = tuple(sorted(index))
        sums[term] = sums.get(term, np.zeros(tensor.shape[degree:])) + tensor[index]
        if np.any(tensor[index] != 0.0):
            touched.add(term)
    return {term: sums[term] for term in sorted(touched)}


def test_tuple_projection_matches_brute_force_and_polynomial_value() -> None:
    generator = np.random.default_rng(17)
    tensor = generator.normal(size=(4, 4, 4, 2))
    for index in itertools.permutations((0, 1, 3)):
        tensor[index] = 0.0
    tensor[2, 2, 2] = 0.0
    # Cancelling nonzero entries keep their term with an exactly zero coefficient.
    for index in itertools.permutations((1, 2, 3)):
        tensor[index] = 0.0
    tensor[1, 2, 3] = 1.5
    tensor[3, 2, 1] = -1.5

    terms, coefficients = project_tuple_coefficients(tensor, 4, 3, maximum_entries=128)
    expected = _brute_force_projection(tensor, 4, 3)

    assert terms == tuple(expected)
    assert (0, 1, 3) not in terms
    assert (2, 2, 2) not in terms
    np.testing.assert_array_equal(coefficients[terms.index((1, 2, 3))], [0.0, 0.0])
    np.testing.assert_allclose(
        coefficients, np.stack(list(expected.values())), rtol=1e-14, atol=1e-15
    )

    plan = ProductGraphPlan(4, terms, maximum_nodes=64)
    points = generator.normal(size=(6, 4))
    direct = sum(
        tensor[index][None, :] * np.prod(points[:, list(index)], axis=1)[:, None]
        for index in itertools.product(range(4), repeat=3)
    )
    np.testing.assert_allclose(
        plan.contract(jnp.asarray(points), jnp.asarray(coefficients)),
        direct,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("build", "error", "message"),
    (
        (lambda: ProductGraphPlan(3, (), maximum_nodes=4), ValueError, "at least one"),
        (lambda: ProductGraphPlan(3, ((0, 3),), maximum_nodes=4), ValueError, "outside"),
        (
            lambda: ProductGraphPlan(3, ((0, 1, 2),), maximum_nodes=8, maximum_degree=2),
            ValueError,
            "maximum_degree=2",
        ),
        (
            lambda: ProductGraphPlan(3, ((0,),), maximum_nodes=4).term_index((1,)),
            ValueError,
            "not a term",
        ),
        (
            lambda: ProductGraphPlan(3, ((0,),), maximum_nodes=4).evaluate(
                np.ones((2, 3), dtype=np.int32)
            ),
            TypeError,
            "variables",
        ),
        (
            lambda: ProductGraphPlan(3, ((0,),), maximum_nodes=4).evaluate(
                np.ones((2, 4))
            ),
            ValueError,
            "variables",
        ),
        (
            lambda: ProductGraphPlan(3, ((0,),), maximum_nodes=4).evaluate(
                np.ones((2, 3)), lane_tile=0
            ),
            ValueError,
            "lane_tile",
        ),
        (
            lambda: project_tuple_coefficients(
                np.ones((4, 4, 4, 3)), 4, 3, maximum_entries=191
            ),
            ValueError,
            "192 entries",
        ),
        (
            lambda: project_tuple_coefficients(
                np.ones((4, 4, 3)), 4, 3, maximum_entries=10**3
            ),
            ValueError,
            "leading shape",
        ),
    ),
    ids=(
        "empty-requests",
        "index-range",
        "degree-bound",
        "absent-term",
        "integer-variables",
        "variable-count",
        "lane-tile",
        "entry-budget",
        "tensor-shape",
    ),
)
def test_invalid_requests_are_refused(
    build: Callable[[], object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        build()

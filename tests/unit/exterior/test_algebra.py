"""Independent numerical contracts for the exterior coefficient kernel."""

from itertools import permutations
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.exterior import (
    exterior_derivative_from_jacobian,
    exterior_indices,
    form_to_vector,
    FormType,
    FormValueSpec,
    hodge_star,
    inner,
    interior,
    map_reference_values,
    pullback,
    to_twisted,
    to_untwisted,
    vector_to_form,
    wedge,
    wedge_sign,
)


pytestmark = pytest.mark.strict_jax


@pytest.mark.parametrize(
    "dimension,degree", [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (4, 2)]
)
def test_basis_order_and_permutation_sign(dimension: int, degree: int) -> None:
    basis = exterior_indices(dimension, degree)
    assert basis == tuple(
        sorted(set(tuple(sorted(p)) for p in permutations(range(dimension), degree)))
    )
    for blade in basis:
        complement = tuple(a for a in range(dimension) if a not in blade)
        merged = blade + complement
        matrix = np.eye(dimension, dtype=np.float64)[list(merged)]
        assert wedge_sign(blade, complement) == round(np.linalg.det(matrix))
    assert len(basis) == factorial(dimension) // (
        factorial(degree) * factorial(dimension - degree)
    )


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("negative_index", [0, 1])
@pytest.mark.parametrize("twist", ["untwisted", "twisted"])
def test_hodge_square_and_volume_pairing(
    degree: int, negative_index: int, twist: str
) -> None:
    from phydrax.exterior import FormTwist
    from phydrax.typing import parse

    form_type = FormType(3, degree, twist=parse(twist, FormTwist, "twist"))
    diagonal = jnp.asarray(
        [(-2.0 if negative_index else 2.0), 3.0, 5.0], dtype=jnp.float64
    )
    inverse = jnp.diag(1.0 / diagonal)
    volume = jnp.sqrt(jnp.asarray(30.0, dtype=jnp.float64))
    values = jnp.arange(1, form_type.component_count + 1, dtype=jnp.float64)
    dual_type = form_type.hodge_dual()
    dual = jax.jit(lambda v: hodge_star(v, form_type, inverse, volume))(values)
    actual = hodge_star(dual, dual_type, inverse, volume)
    sign = (-1) ** (degree * (3 - degree) + negative_index)
    np.testing.assert_allclose(actual, sign * values, atol=1e-12)
    pairing = wedge(values, dual, form_type, dual_type)
    np.testing.assert_allclose(
        pairing,
        jnp.reshape(inner(values, values, form_type, inverse) * volume, (1,)),
        atol=1e-12,
    )


def test_matrix_wedge_preserves_product_order_and_batch_axis() -> None:
    form_type = FormType(2, 1, fiber_shape=(2, 2))
    a = jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64)
    b = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    zero = jnp.zeros((2, 2), dtype=jnp.float64)
    left, right = jnp.stack((a, zero)), jnp.stack((zero, b))
    actual = jax.jit(lambda l, r: wedge(l, r, form_type, form_type, product="matrix"))(
        left[None, ...], right
    )
    np.testing.assert_allclose(actual, (a @ b)[None, None, ...])
    reverse = wedge(right, left, form_type, form_type, product="matrix")
    np.testing.assert_allclose(reverse, -(b @ a)[None, ...])
    assert not np.array_equal(actual[0], -reverse)


def _polynomial_one_form(point: Array) -> Array:
    x, y, z = point
    return jnp.stack((x * y * z, x * x * z, x * y * y))


def test_polynomial_exterior_derivative_and_nilpotency_are_differentiable() -> None:
    one = FormType(3, 1, twist="twisted")
    two = one.exterior_derivative_type()
    point = jnp.asarray([2.0, 3.0, 5.0], dtype=jnp.float64)

    def derivative(q: Array) -> Array:
        return exterior_derivative_from_jacobian(jax.jacfwd(_polynomial_one_form)(q), one)

    actual = jax.jit(derivative)(point)
    x, y, z = point
    expected = jnp.stack((x * z, y * y - x * y, 2 * x * y - x * x))
    np.testing.assert_allclose(actual, expected)
    twice = exterior_derivative_from_jacobian(jax.jacfwd(derivative)(point), two)
    np.testing.assert_allclose(twice, jnp.zeros((1,), dtype=jnp.float64), atol=1e-12)
    expected_gradient = jnp.asarray(
        [[5.0, 0.0, 2.0], [-3.0, 4.0, 0.0], [2.0, 4.0, 0.0]], dtype=jnp.float64
    )
    np.testing.assert_allclose(jax.jacrev(derivative)(point), expected_gradient)


def test_interior_square_vanishes_with_matrix_fibers_and_broadcast_batch() -> None:
    form_type = FormType(3, 2, fiber_shape=(2, 2))
    values = jnp.arange(12, dtype=jnp.float64).reshape((1, 3, 2, 2))
    vector = jnp.asarray([2.0, 3.0, -1.0], dtype=jnp.float64)
    first = interior(vector, values, form_type)
    twice = interior(vector, first, form_type.interior_type())
    np.testing.assert_allclose(
        twice, jnp.zeros((1, 1, 2, 2), dtype=jnp.float64), atol=1e-12
    )


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_twisted_reflection_is_orientation_line_action(degree: int) -> None:
    ordinary = FormType(2, degree)
    twisted = ordinary.with_twist("twisted")
    values = jnp.arange(1, ordinary.component_count + 1, dtype=jnp.float64)
    reflection = jnp.diag(jnp.asarray([-1.0, 1.0], dtype=jnp.float64))
    result = jax.jit(lambda v: pullback(v, twisted, reflection))(values[None, ...])
    np.testing.assert_allclose(result, -pullback(values[None, ...], ordinary, reflection))
    np.testing.assert_allclose(
        to_untwisted(to_twisted(values, ordinary, -1), twisted, -1), values
    )


def test_nonsquare_twisted_pullback_requires_declared_coorientation() -> None:
    form_type = FormType(3, 1, twist="twisted")
    jacobian = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64)
    values = jnp.asarray([2.0, 3.0, 7.0], dtype=jnp.float64)
    with pytest.raises(ValueError, match="coorientation"):
        pullback(values, form_type, jacobian)
    np.testing.assert_allclose(
        pullback(values, form_type, jacobian, coorientation=-1),
        jnp.asarray([-2.0, -3.0], dtype=jnp.float64),
    )


@pytest.mark.parametrize("dimension,expected", [(2, [-3.0, 2.0]), (3, [5.0, -3.0, 2.0])])
def test_flux_proxy_is_outward_normal_first(
    dimension: int, expected: list[float]
) -> None:
    spec = FormValueSpec(
        FormType(dimension, dimension - 1, twist="twisted"), proxy="flux"
    )
    vector = jnp.asarray([2.0, 3.0, 5.0][:dimension], dtype=jnp.float64)
    coefficients = vector_to_form(vector, spec)
    np.testing.assert_array_equal(coefficients, jnp.asarray(expected, dtype=jnp.float64))
    np.testing.assert_array_equal(
        form_to_vector(coefficients[None, ...], spec), vector[None, ...]
    )


def test_single_sample_scalar_proxy_does_not_drop_batch_axis() -> None:
    spec = FormValueSpec(FormType(2, 0), proxy="scalar")
    values = jnp.asarray([7.0], dtype=jnp.float64)
    coefficients = vector_to_form(values, spec)
    assert coefficients.shape == (1, 1)
    np.testing.assert_array_equal(form_to_vector(coefficients, spec), values)


@pytest.mark.parametrize(
    "twist,expected", [("untwisted", [-2.0, 2.0]), ("twisted", [2.0, -2.0])]
)
def test_reflected_flux_piola_determinant_law(twist: str, expected: list[float]) -> None:
    from phydrax.exterior import FormTwist
    from phydrax.typing import parse

    spec = FormValueSpec(
        FormType(2, 1, twist=parse(twist, FormTwist, "twist")), proxy="flux"
    )
    jacobian = jnp.diag(jnp.asarray([2.0, -3.0], dtype=jnp.float64))
    values = jnp.asarray([6.0, 4.0], dtype=jnp.float64)
    np.testing.assert_allclose(
        map_reference_values(values, spec, jacobian),
        jnp.asarray(expected, dtype=jnp.float64),
    )


def test_embedded_covariant_piola_preserves_tangent_pairing() -> None:
    spec = FormValueSpec(FormType(2, 1), proxy="circulation")
    jacobian = jnp.asarray([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=jnp.float64)
    reference = jnp.asarray([4.0, -2.0], dtype=jnp.float64)
    mapped = jax.jit(lambda j: map_reference_values(reference, spec, j))(jacobian)
    np.testing.assert_allclose(jacobian.T @ mapped, reference, atol=1e-12)
    normal = jnp.cross(jacobian[:, 0], jacobian[:, 1])
    np.testing.assert_allclose(
        normal @ mapped, jnp.asarray(0.0, dtype=jnp.float64), atol=1e-12
    )
    tangent = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    np.testing.assert_allclose(mapped @ (jacobian @ tangent), reference @ tangent)


def test_form_type_persistence_and_scientific_identity() -> None:
    value = FormType(2, 1, twist="twisted", fiber_shape=(2, 3), ambient_dimension=3)
    restored = FormType.from_dict(value.to_dict())
    assert restored == value
    assert hash(restored) == hash(value)
    assert restored.value_shape == (3, 2, 3)
    assert restored != value.with_twist("untwisted")
    spec = FormValueSpec(value, proxy="components")
    assert FormValueSpec.from_dict(spec.to_dict()) == spec


def test_hodge_dual_refuses_an_embedded_coefficient_frame() -> None:
    with pytest.raises(ValueError, match="Hodge"):
        FormType(2, 1, ambient_dimension=3).hodge_dual()


def test_density_proxy_refuses_a_nontop_degree() -> None:
    with pytest.raises(ValueError, match="degree"):
        FormValueSpec(FormType(3, 1), proxy="density")


def test_matrix_coefficients_require_a_declared_matrix_product() -> None:
    form_type = FormType(2, 1, fiber_shape=(2, 2))
    values = jnp.ones((2, 2, 2), dtype=jnp.float64)
    with pytest.raises(ValueError, match="Scalar"):
        wedge(values, values, form_type, form_type)


@pytest.mark.parametrize("degrees", [(1, 1, 1), (1, 2, 1), (0, 1, 2)])
def test_scalar_wedge_associativity_and_graded_commutativity(
    degrees: tuple[int, int, int],
) -> None:
    a_type, b_type, c_type = (FormType(4, degree) for degree in degrees)
    a = jnp.arange(1, a_type.component_count + 1, dtype=jnp.float64)
    b = jnp.arange(b_type.component_count, 0, -1, dtype=jnp.float64)
    c = jnp.arange(2, c_type.component_count + 2, dtype=jnp.float64)
    ab = wedge(a, b, a_type, b_type)
    bc = wedge(b, c, b_type, c_type)
    left = wedge(ab, c, a_type.wedge_type(b_type), c_type)
    right = wedge(a, bc, a_type, b_type.wedge_type(c_type))
    np.testing.assert_allclose(left, right)
    np.testing.assert_allclose(
        ab, ((-1) ** (degrees[0] * degrees[1])) * wedge(b, a, b_type, a_type)
    )


def test_rectangular_pullback_naturality_matches_explicit_one_form_products() -> None:
    form_type = FormType(4, 1)
    a = jnp.asarray([2.0, 3.0, -1.0, 5.0], dtype=jnp.float64)
    b = jnp.asarray([-1.0, 7.0, 4.0, 2.0], dtype=jnp.float64)
    jacobian = jnp.asarray(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, -1.0], [1.0, 1.0, 1.0]],
        dtype=jnp.float64,
    )
    actual = pullback(
        wedge(a, b, form_type, form_type), form_type.wedge_type(form_type), jacobian
    )
    pulled_a, pulled_b = jacobian.T @ a, jacobian.T @ b
    expected = jnp.stack(
        (
            pulled_a[0] * pulled_b[1] - pulled_a[1] * pulled_b[0],
            pulled_a[0] * pulled_b[2] - pulled_a[2] * pulled_b[0],
            pulled_a[1] * pulled_b[2] - pulled_a[2] * pulled_b[1],
        )
    )
    np.testing.assert_allclose(actual, expected)
    source_type = FormType(3, 1)
    np.testing.assert_allclose(
        actual,
        wedge(
            pullback(a, form_type, jacobian),
            pullback(b, form_type, jacobian),
            source_type,
            source_type,
        ),
    )


@pytest.mark.parametrize("complex_coefficients", [False, True])
def test_n_dimensional_covariant_map_uses_native_solve_and_differentiates(
    complex_coefficients: bool,
) -> None:
    spec = FormValueSpec(FormType(5, 1), proxy="circulation")
    vector = jnp.arange(1, 6, dtype=jnp.float64)
    if complex_coefficients:
        vector = vector.astype(jnp.complex128) * (1.0 + 2.0j)
    diagonal = jnp.asarray([2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float64)

    def mapped(scale: Array) -> Array:
        return map_reference_values(vector, spec, jnp.diag(scale))

    reference_scale = diagonal.astype(vector.dtype)
    np.testing.assert_allclose(
        jax.jit(mapped)(diagonal), vector / reference_scale, atol=1e-12
    )
    np.testing.assert_allclose(
        jax.jacfwd(mapped)(diagonal),
        jnp.diag(-vector / (reference_scale * reference_scale)),
        atol=1e-12,
    )


def test_complex_hodge_pairing_is_hermitian_over_real_metric() -> None:
    form_type = FormType(2, 1)
    values = jnp.asarray([1.0 + 2.0j, 3.0 - 4.0j], dtype=jnp.complex128)
    inverse = jnp.diag(jnp.asarray([0.5, 1.0 / 3.0], dtype=jnp.float64))
    expected = jnp.asarray(5.0 / 2.0 + 25.0 / 3.0, dtype=jnp.complex128)
    np.testing.assert_allclose(
        jax.jit(lambda a: inner(a, a, form_type, inverse))(values), expected, atol=1e-12
    )
    dual = hodge_star(
        values, form_type, inverse, jnp.sqrt(jnp.asarray(6.0, dtype=jnp.float64))
    )
    np.testing.assert_allclose(
        hodge_star(
            dual,
            form_type.hodge_dual(),
            inverse,
            jnp.sqrt(jnp.asarray(6.0, dtype=jnp.float64)),
        ),
        -values,
        atol=1e-12,
    )


def test_batched_metric_pairing_preserves_independent_metric_samples() -> None:
    form_type = FormType(2, 1)
    values = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    inverse = jnp.stack(
        (jnp.eye(2, dtype=jnp.float64), 2.0 * jnp.eye(2, dtype=jnp.float64))
    )
    volume = jnp.asarray([1.0, 0.5], dtype=jnp.float64)
    np.testing.assert_allclose(
        inner(values, values, form_type, inverse),
        jnp.asarray([13.0, 26.0], dtype=jnp.float64),
    )
    dual = hodge_star(values, form_type, inverse, volume)
    np.testing.assert_allclose(
        hodge_star(dual, form_type.hodge_dual(), inverse, volume),
        jnp.stack((-values, -values)),
    )


def test_ambiguous_two_dimensional_degree_uses_declared_proxy_map() -> None:
    form_type = FormType(2, 1)
    circulation = FormValueSpec(form_type, proxy="circulation")
    flux = FormValueSpec(form_type, proxy="flux")
    jacobian = jnp.diag(jnp.asarray([2.0, -3.0], dtype=jnp.float64))
    values = jnp.asarray([6.0, 4.0], dtype=jnp.float64)
    np.testing.assert_allclose(
        map_reference_values(values, circulation, jacobian),
        jnp.asarray([3.0, -4.0 / 3.0], dtype=jnp.float64),
    )
    np.testing.assert_allclose(
        map_reference_values(values, flux, jacobian),
        jnp.asarray([-2.0, 2.0], dtype=jnp.float64),
    )
    assert circulation.value_spec_id != flux.value_spec_id


@pytest.mark.parametrize(
    "dimension,degree,ambient", [(2, 3, None), (3, 1, 2), (2, -1, None), (-1, 0, None)]
)
def test_form_type_refuses_invalid_dimension_degree_domains(
    dimension: int, degree: int, ambient: int | None
) -> None:
    with pytest.raises(ValueError):
        FormType(dimension, degree, ambient_dimension=ambient)


def test_form_twist_refuses_a_proxy_word() -> None:
    with pytest.raises(ValueError):
        FormType(3, 1, twist="density")  # ty: ignore[invalid-argument-type]


def test_proxy_refuses_a_conformity_word() -> None:
    with pytest.raises(ValueError):
        FormValueSpec(FormType(3, 1), proxy="hdiv")  # ty: ignore[invalid-argument-type]


def test_fiber_product_refuses_an_undeclared_coefficient_algebra() -> None:
    with pytest.raises(ValueError):
        wedge(
            jnp.ones((3,), dtype=jnp.float64),
            jnp.ones((3,), dtype=jnp.float64),
            FormType(3, 1),
            FormType(3, 1),
            product="tensor",  # ty: ignore[invalid-argument-type]
        )


def test_hodge_geometry_parameter_gradients_match_anisotropic_metric_formula() -> None:
    form_type = FormType(2, 1)
    values = jnp.asarray([1.0, 2.0], dtype=jnp.float64)

    def star(scale: Array) -> Array:
        inverse = jnp.diag(jnp.stack((1.0 / (scale * scale), jnp.ones_like(scale))))
        return hodge_star(values, form_type, inverse, scale)

    scale = jnp.asarray(0.7, dtype=jnp.float64)
    np.testing.assert_allclose(
        jax.jit(star)(scale), jnp.stack((-2.0 * scale, 1.0 / scale)), atol=1e-12
    )
    np.testing.assert_allclose(
        jax.jacfwd(star)(scale),
        jnp.stack((jnp.asarray(-2.0, dtype=jnp.float64), -1.0 / (scale * scale))),
        atol=1e-12,
    )

"""Traceable exterior algebra with a stable coefficient axis and explicit fibers."""

from __future__ import annotations

from math import prod
from numbers import Integral

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._validation import nonnegative_integer
from ..ein import contract
from ..linalg._compound import compound_matrix
from ..linalg._operators import DenseLinearOperator
from ..linalg._policies import DenseLU, LinearDerivativeSolvePolicy, LinearSolvePolicy
from ..linalg._problems import LinearSystem
from ..linalg._runtime import solve
from ..linalg._small_batched import SmallLinearSolvePlan, solve_small_linear
from ..linalg._spaces import ArraySpace, RHSLayout
from ._basis import complement_table, derivative_table, interior_table, wedge_table
from ._form_type import FiberProduct, FormType, FormValueSpec


def _form_array(values: ArrayLike, form_type: FormType, /) -> Array:
    array = jnp.asarray(values)
    shape = form_type.value_shape
    if array.ndim < len(shape) or array.shape[-len(shape) :] != shape:
        raise ValueError(
            f"Form coefficients require trailing shape {shape}; got {array.shape}."
        )
    return array


def _promote_factors(left: Array, right: Array, /) -> tuple[Array, Array]:
    dtype = np.promote_types(left.dtype, right.dtype)
    return (
        left if left.dtype == dtype else left.astype(dtype),
        right if right.dtype == dtype else right.astype(dtype),
    )


def _broadcast_factors(left: Array, right: Array, /) -> tuple[Array, Array]:
    left, right = _promote_factors(left, right)
    rank = max(left.ndim, right.ndim)
    return (
        left.reshape((*(1 for _ in range(rank - left.ndim)), *left.shape)),
        right.reshape((*(1 for _ in range(rank - right.ndim)), *right.shape)),
    )


def _multiply(left: Array, right: Array, /) -> Array:
    a, b = _broadcast_factors(left, right)
    return a * b


def _matrix_multiply(left: Array, right: Array, /) -> Array:
    left, right = _promote_factors(left, right)
    batch_shape = jnp.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    # One batch axis avoids a compiler rewrite of transposed dot batch axes.
    # Keep the fiber contraction bounded by the actual matrix product, without
    # materializing an outer product over matrix rows and columns.
    batch_count = prod(batch_shape)
    a = jnp.broadcast_to(left, (*batch_shape, *left.shape[-2:])).reshape(
        (batch_count, *left.shape[-2:])
    )
    b = jnp.broadcast_to(right, (*batch_shape, *right.shape[-2:])).reshape(
        (batch_count, *right.shape[-2:])
    )
    result = contract("bij,bjk->bik", a, b, backend="jax")
    return result.reshape((*batch_shape, left.shape[-2], right.shape[-1]))


def _divide(left: Array, right: Array, /) -> Array:
    a, b = _broadcast_factors(left, right)
    return a / b


def _scatter(
    terms: Array, output: tuple[int, ...], count: int, fiber_rank: int, /
) -> Array:
    axis = terms.ndim - fiber_rank - 1
    moved = jnp.moveaxis(terms, axis, 0)
    result = (
        jnp.zeros((count, *moved.shape[1:]), dtype=terms.dtype)
        .at[jnp.asarray(output, dtype=jnp.int32)]
        .add(moved)
    )
    return jnp.moveaxis(result, 0, axis)


def _signs(signs: tuple[int, ...], dtype: DTypeLike, fiber_rank: int, /) -> Array:
    return jnp.asarray(signs, dtype=dtype).reshape(
        (len(signs), *(1 for _ in range(fiber_rank)))
    )


def wedge(
    left: ArrayLike,
    right: ArrayLike,
    left_type: FormType,
    right_type: FormType,
    /,
    *,
    product: FiberProduct = "scalar",
) -> Array:
    result_type = left_type.wedge_type(right_type, product=product)
    a, b = _form_array(left, left_type), _form_array(right, right_type)
    li, ri, output, signs = wedge_table(
        left_type.ambient_dimension, left_type.degree, right_type.degree
    )
    lterms = jnp.take(
        a, jnp.asarray(li, dtype=jnp.int32), axis=-len(left_type.fiber_shape) - 1
    )
    rterms = jnp.take(
        b, jnp.asarray(ri, dtype=jnp.int32), axis=-len(right_type.fiber_shape) - 1
    )
    match product:
        case "scalar":
            terms = _multiply(lterms, rterms)
        case "matrix":
            terms = _matrix_multiply(lterms, rterms)
        case _:
            raise ValueError("Invalid fiber product.")
    sign = _signs(signs, terms.dtype, len(result_type.fiber_shape))
    terms = terms * sign.reshape(
        (*(1 for _ in range(terms.ndim - sign.ndim)), *sign.shape)
    )
    return _scatter(
        terms, output, result_type.component_count, len(result_type.fiber_shape)
    )


def exterior_derivative_from_jacobian(
    jacobian: ArrayLike, form_type: FormType, /
) -> Array:
    result_type = form_type.exterior_derivative_type()
    derivative = jnp.asarray(jacobian)
    expected = (*form_type.value_shape, form_type.ambient_dimension)
    if derivative.ndim < len(expected) or derivative.shape[-len(expected) :] != expected:
        raise ValueError(f"Form Jacobian requires trailing shape {expected}.")
    source, axes, output, signs = derivative_table(
        form_type.ambient_dimension, form_type.degree
    )
    moved = jnp.moveaxis(derivative, -len(form_type.fiber_shape) - 2, -2)
    terms = moved[
        ..., jnp.asarray(source, dtype=jnp.int32), jnp.asarray(axes, dtype=jnp.int32)
    ]
    terms = jnp.moveaxis(terms, -1, -len(form_type.fiber_shape) - 1)
    sign = _signs(signs, terms.dtype, len(form_type.fiber_shape))
    terms = terms * sign.reshape(
        (*(1 for _ in range(terms.ndim - sign.ndim)), *sign.shape)
    )
    return _scatter(
        terms, output, result_type.component_count, len(form_type.fiber_shape)
    )


def interior(vector: ArrayLike, form: ArrayLike, form_type: FormType, /) -> Array:
    result_type = form_type.interior_type()
    values, vectors = _form_array(form, form_type), jnp.asarray(vector)
    if vectors.shape[-1:] != (form_type.ambient_dimension,):
        raise ValueError("Interior vector dimension does not match ambient dimension.")
    axes, source, output, signs = interior_table(
        form_type.ambient_dimension, form_type.degree
    )
    terms = jnp.take(
        values, jnp.asarray(source, dtype=jnp.int32), axis=-len(form_type.fiber_shape) - 1
    )
    vector_terms = vectors[..., jnp.asarray(axes, dtype=jnp.int32)]
    vector_terms = vector_terms.reshape(
        (*vector_terms.shape, *(1 for _ in form_type.fiber_shape))
    )
    terms = _multiply(terms, vector_terms)
    sign = _signs(signs, terms.dtype, len(form_type.fiber_shape))
    terms = terms * sign.reshape(
        (*(1 for _ in range(terms.ndim - sign.ndim)), *sign.shape)
    )
    return _scatter(
        terms, output, result_type.component_count, len(form_type.fiber_shape)
    )


def _linear_components(matrix: Array, form: Array, form_type: FormType, /) -> Array:
    rank = len(form_type.fiber_shape)
    batch = form.shape[: -rank - 1]
    flat = form.reshape((*batch, form_type.component_count, prod(form_type.fiber_shape)))
    matrix, flat = _promote_factors(matrix, flat)
    result = contract("...ij,...jf->...if", matrix, flat, backend="jax")
    return result.reshape((*result.shape[:-2], result.shape[-2], *form_type.fiber_shape))


def hodge_star(
    form: ArrayLike,
    form_type: FormType,
    inverse_metric: ArrayLike,
    volume_density: ArrayLike,
    /,
) -> Array:
    result_type = form_type.hodge_dual()
    values = _form_array(form, form_type)
    inverse = jnp.asarray(inverse_metric)
    if inverse.shape[-2:] != (form_type.dimension, form_type.dimension):
        raise ValueError("Inverse metric dimension does not match form dimension.")
    paired = _linear_components(
        compound_matrix(inverse, form_type.degree), values, form_type
    )
    output, signs = complement_table(form_type.dimension, form_type.degree)
    rank = len(form_type.fiber_shape)
    sign = _signs(signs, paired.dtype, rank)
    density = jnp.asarray(volume_density).reshape(
        (*jnp.shape(volume_density), *(1 for _ in range(rank + 1)))
    )
    terms = _multiply(
        paired
        * sign.reshape((*(1 for _ in range(paired.ndim - sign.ndim)), *sign.shape)),
        density,
    )
    return _scatter(terms, output, result_type.component_count, rank)


def inner(
    left: ArrayLike, right: ArrayLike, form_type: FormType, inverse_metric: ArrayLike, /
) -> Array:
    a, b = _form_array(left, form_type), _form_array(right, form_type)
    inverse = jnp.asarray(inverse_metric)
    if inverse.shape[-2:] != (form_type.ambient_dimension, form_type.ambient_dimension):
        raise ValueError("Inverse metric dimension does not match form coordinates.")
    paired = _linear_components(compound_matrix(inverse, form_type.degree), b, form_type)
    contracted = _multiply(jnp.conj(a), paired)
    return jnp.sum(
        contracted,
        axis=tuple(
            range(contracted.ndim - len(form_type.fiber_shape) - 1, contracted.ndim)
        ),
    )


def _coorientation(value: int | None, /) -> int:
    if value is None:
        raise ValueError("A non-square twisted map requires coorientation +1 or -1.")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Orientation must be a static integer.")
    if value not in (-1, 1):
        raise ValueError("Orientation must be +1 or -1.")
    return int(value)


def pullback(
    form: ArrayLike,
    form_type: FormType,
    jacobian: ArrayLike,
    /,
    *,
    coorientation: int | None = None,
) -> Array:
    values, jac = _form_array(form, form_type), jnp.asarray(jacobian)
    if jac.ndim < 2 or jac.shape[-2] != form_type.ambient_dimension:
        raise ValueError(
            "Pullback Jacobian target dimension does not match coefficients."
        )
    if form_type.degree > jac.shape[-1]:
        raise ValueError("Pullback degree exceeds source dimension.")
    transformed = _linear_components(
        compound_matrix(jnp.swapaxes(jac, -1, -2), form_type.degree), values, form_type
    )
    if form_type.twist == "twisted":
        sign = (
            jnp.sign(compound_matrix(jac, jac.shape[-1])[..., 0, 0])
            if jac.shape[-2] == jac.shape[-1]
            else jnp.asarray(_coorientation(coorientation), dtype=jac.dtype)
        )
        transformed = _multiply(
            transformed,
            sign.reshape(
                (*sign.shape, *(1 for _ in range(len(form_type.fiber_shape) + 1)))
            ),
        )
    return transformed


def to_untwisted(form: ArrayLike, form_type: FormType, orientation: int, /) -> Array:
    if form_type.twist != "twisted":
        raise ValueError("to_untwisted requires a twisted form.")
    return _form_array(form, form_type) * _coorientation(orientation)


def to_twisted(form: ArrayLike, form_type: FormType, orientation: int, /) -> Array:
    if form_type.twist != "untwisted":
        raise ValueError("to_twisted requires an untwisted form.")
    return _form_array(form, form_type) * _coorientation(orientation)


def hodge_square_sign(form_type: FormType, negative_index: int = 0, /) -> int:
    negative_index = nonnegative_integer(negative_index, "negative_index")
    if not 0 <= negative_index <= form_type.dimension:
        raise ValueError("Negative index must lie in [0,dimension].")
    return (-1) ** (
        form_type.degree * (form_type.dimension - form_type.degree) + negative_index
    )


def codifferential_sign(form_type: FormType, negative_index: int = 0, /) -> int:
    form_type.codifferential_type()
    negative_index = nonnegative_integer(negative_index, "negative_index")
    if not 0 <= negative_index <= form_type.dimension:
        raise ValueError("Negative index must lie in [0,dimension].")
    return (-1) ** (form_type.dimension * (form_type.degree + 1) + negative_index + 1)


def _proxy_array(values: ArrayLike, value_spec: FormValueSpec, /) -> Array:
    array = jnp.asarray(values)
    shape = value_spec.value_shape
    if array.ndim < len(shape) or (shape and array.shape[-len(shape) :] != shape):
        raise ValueError(f"Proxy values require trailing shape {shape}.")
    return array


def vector_to_form(values: ArrayLike, value_spec: FormValueSpec, /) -> Array:
    array = _proxy_array(values, value_spec)
    form_type = value_spec.form_type
    rank = len(form_type.fiber_shape)
    match value_spec.proxy:
        case "components" | "circulation":
            return array
        case "scalar" | "density":
            if form_type.component_count != 1:
                raise ValueError(
                    "Embedded density requires exterior component coordinates."
                )
            return jnp.expand_dims(array, array.ndim - rank)
        case "flux":
            if form_type.dimension != form_type.ambient_dimension:
                raise ValueError(
                    "Embedded flux coefficients require a tangent frame/coorientation map."
                )
            axes, _, _, signs = interior_table(form_type.dimension, form_type.dimension)
            result = jnp.take(array, jnp.asarray(axes, dtype=jnp.int32), axis=-rank - 1)
            sign = _signs(signs, result.dtype, rank)
            return result * sign.reshape(
                (*(1 for _ in range(result.ndim - sign.ndim)), *sign.shape)
            )


def form_to_vector(form: ArrayLike, value_spec: FormValueSpec, /) -> Array:
    values = _form_array(form, value_spec.form_type)
    rank = len(value_spec.form_type.fiber_shape)
    match value_spec.proxy:
        case "components" | "circulation":
            return values
        case "scalar" | "density":
            if value_spec.form_type.component_count != 1:
                raise ValueError(
                    "Embedded density requires exterior component coordinates."
                )
            return jnp.squeeze(values, axis=-rank - 1)
        case "flux":
            form_type = value_spec.form_type
            if form_type.dimension != form_type.ambient_dimension:
                raise ValueError(
                    "Embedded flux coefficients require a tangent frame/coorientation map."
                )
            axes, _, _, signs = interior_table(form_type.dimension, form_type.dimension)
            routes = tuple(axes.index(axis) for axis in range(form_type.dimension))
            result = jnp.take(
                values, jnp.asarray(routes, dtype=jnp.int32), axis=-rank - 1
            )
            sign = _signs(tuple(signs[route] for route in routes), result.dtype, rank)
            return result * sign.reshape(
                (*(1 for _ in range(result.ndim - sign.ndim)), *sign.shape)
            )


def _solve_mapping(matrix: Array, rhs: Array, /) -> Array:
    """Apply the owning native solve without numeric-derived static identities."""
    matrix, rhs = _promote_factors(matrix, rhs)
    dtype = matrix.dtype
    n = matrix.shape[-1]
    if n <= 4:
        result = solve_small_linear(SmallLinearSolvePlan(n), matrix, rhs)
        return jnp.where(
            result.successful[..., None, None],
            result.value,
            jnp.asarray(jnp.nan, dtype=result.value.dtype),
        )
    space = ArraySpace(
        (n,), dtype=dtype, space_id=f"exterior_reference_coefficients:{n}:{dtype}"
    )
    operator = DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        operator_id=f"exterior_coordinate_mapping:{n}:{dtype}",
    )
    result = solve(
        LinearSystem(operator),
        rhs,
        policy=LinearSolvePolicy(
            DenseLU(),
            derivative_solve=LinearDerivativeSolvePolicy(route="primal-factors"),
        ),
        rhs_layout=RHSLayout((rhs.shape[-1],)),
    )
    value = jnp.asarray(result.value)
    successful = jnp.all(result.successful, axis=-1)
    return jnp.where(
        successful[..., None, None], value, jnp.asarray(jnp.nan, dtype=value.dtype)
    )


def _solve_components(matrix: Array, values: Array, form_type: FormType, /) -> Array:
    rank = len(form_type.fiber_shape)
    batch = jnp.broadcast_shapes(matrix.shape[:-2], values.shape[: -rank - 1])
    matrix_batch = (1,) * (len(batch) - matrix.ndim + 2) + matrix.shape[:-2]
    # Value-only broadcast axes are multiple RHS columns of one mapping, not
    # independent matrices. Factor each geometric mapping only once.
    rhs_axes = tuple(
        axis
        for axis, (matrix_extent, extent) in enumerate(
            zip(matrix_batch, batch, strict=True)
        )
        if matrix_extent == 1 and extent != 1
    )
    retained_axes = tuple(axis for axis in range(len(batch)) if axis not in rhs_axes)
    retained_batch = tuple(batch[axis] for axis in retained_axes)
    rhs_batch = tuple(batch[axis] for axis in rhs_axes)
    matrix = matrix.reshape((*matrix_batch, *matrix.shape[-2:]))
    matrix = jnp.squeeze(matrix, axis=rhs_axes)
    values = jnp.broadcast_to(values, (*batch, *form_type.value_shape))
    fiber_count = prod(form_type.fiber_shape)
    values = values.reshape((*batch, form_type.component_count, fiber_count))
    order = (*retained_axes, len(batch), *rhs_axes, len(batch) + 1)
    rhs = values.transpose(order).reshape(
        (*retained_batch, form_type.component_count, prod(rhs_batch) * fiber_count)
    )
    result = _solve_mapping(matrix, rhs).reshape(
        (*retained_batch, form_type.component_count, *rhs_batch, fiber_count)
    )
    inverse_order = tuple(order.index(axis) for axis in range(len(order)))
    return result.transpose(inverse_order).reshape((*batch, *form_type.value_shape))


def _mapping_determinant(
    jacobian: Array, form_type: FormType, coorientation: int | None, /
) -> Array:
    n = jacobian.shape[-1]
    if jacobian.shape[-2] == n:
        signed = compound_matrix(jacobian, n)[..., 0, 0]
        return jnp.abs(signed) if form_type.twist == "twisted" else signed
    gram = jnp.swapaxes(jacobian, -1, -2) @ jacobian
    measure = jnp.sqrt(jnp.abs(compound_matrix(gram, n)[..., 0, 0]))
    return (
        measure
        if form_type.twist == "twisted"
        else measure * _coorientation(coorientation)
    )


def _mapping_twist(
    values: Array,
    jacobian: Array,
    form_type: FormType,
    coorientation: int | None,
    event_rank: int,
    /,
) -> Array:
    if form_type.twist == "untwisted":
        return values
    n = jacobian.shape[-1]
    sign = (
        jnp.sign(compound_matrix(jacobian, n)[..., 0, 0])
        if jacobian.shape[-2] == n
        else jnp.asarray(_coorientation(coorientation), dtype=jacobian.dtype)
    )
    return _multiply(values, sign.reshape((*sign.shape, *(1 for _ in range(event_rank)))))


def map_reference_values(
    values: ArrayLike,
    value_spec: FormValueSpec,
    jacobian: ArrayLike,
    /,
    *,
    coorientation: int | None = None,
) -> Array:
    """Push reference proxy values through J (physical rows, reference columns)."""
    array, jac = _proxy_array(values, value_spec), jnp.asarray(jacobian)
    form_type = value_spec.form_type
    if jac.ndim < 2 or jac.shape[-1] != form_type.dimension:
        raise ValueError("Jacobian reference dimension does not match form dimension.")
    if form_type.ambient_dimension != form_type.dimension:
        raise ValueError("Reference form coordinates must be intrinsic.")
    n, m = jac.shape[-1], jac.shape[-2]
    if m < n:
        raise ValueError(
            "Reference-value maps require an immersion (physical dimension >= reference dimension)."
        )
    rank = len(form_type.fiber_shape)
    match value_spec.proxy:
        case "scalar":
            return _mapping_twist(array, jac, form_type, coorientation, rank)
        case "density":
            determinant = _mapping_determinant(jac, form_type, coorientation)
            return _divide(
                array,
                determinant.reshape((*determinant.shape, *(1 for _ in range(rank)))),
            )
        case "flux":
            determinant = _mapping_determinant(jac, form_type, coorientation)
            mapped = _linear_components(jac, array, form_type)
            return _divide(
                mapped,
                determinant.reshape((*determinant.shape, *(1 for _ in range(rank + 1)))),
            )
        case "circulation" | "components":
            if form_type.degree == 0:
                mapped = array
            elif m == n:
                induced = compound_matrix(jnp.swapaxes(jac, -1, -2), form_type.degree)
                mapped = _solve_components(induced, array, form_type)
            else:
                gram = jnp.swapaxes(jac, -1, -2) @ jac
                tangent = _solve_components(
                    compound_matrix(gram, form_type.degree), array, form_type
                )
                mapped = _linear_components(
                    compound_matrix(jac, form_type.degree), tangent, form_type
                )
            return _mapping_twist(mapped, jac, form_type, coorientation, rank + 1)


__all__ = [
    "wedge",
    "interior",
    "exterior_derivative_from_jacobian",
    "hodge_star",
    "inner",
    "pullback",
    "to_twisted",
    "to_untwisted",
    "hodge_square_sign",
    "codifferential_sign",
    "vector_to_form",
    "form_to_vector",
    "map_reference_values",
]

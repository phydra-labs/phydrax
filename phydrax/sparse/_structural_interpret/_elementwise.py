"""Propagation rules for element-wise operations."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax._src.core import JaxprEqn
from jax.core import ShapedArray

from ._common import (
    _atom_const_val,
    _atom_numel,
    _atom_shape,
    _atom_value_bounds,
    _backend_bounds,
    _clear_where_zero,
    _copy_index_sets,
    _empty_index_sets,
    _index_sets,
    _is_floating,
    _numel,
    _propagate_const_binary,
    _store_bounds,
    _store_folded_const,
    _union_elementwise,
    StateBounds,
    StateConsts,
    StateIndices,
)


def _integer_division_defined(dividend: np.ndarray, divisor: np.ndarray) -> bool:
    """Whether integer ``lax.div``/``lax.rem`` have one specified result everywhere.

    XLA leaves division by zero and the signed ``min // -1`` overflow
    implementation defined, so such operands are never folded.
    """
    if not (
        jnp.issubdtype(dividend.dtype, jnp.integer)
        and jnp.issubdtype(divisor.dtype, jnp.integer)
    ):
        return False
    if np.any(divisor == 0):
        return False
    if jnp.issubdtype(dividend.dtype, jnp.signedinteger):
        smallest = np.iinfo(dividend.dtype).min
        return not np.any((dividend == smallest) & (divisor == -1))
    return True


def _integer_remainder(dividend: np.ndarray, divisor: np.ndarray) -> np.ndarray | None:
    """Integer ``lax.rem``: the remainder takes the dividend's sign (C ``%``)."""
    if not _integer_division_defined(dividend, divisor):
        return None
    return np.fmod(dividend, divisor)


# Host evaluators for constant values during tracing. Used to propagate static
# index values through integer arithmetic to gather/scatter; floating results
# are never folded (see ``_store_folded_const``).
_BINARY_CONST_UFUNCS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray | None]] = {
    # arithmetic
    "add": np.add,
    "add_any": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "max": np.maximum,
    "min": np.minimum,
    "rem": _integer_remainder,
    # comparison
    "eq": np.equal,
    "ne": np.not_equal,
    # bitwise
    "and": np.bitwise_and,
    "or": np.bitwise_or,
    "xor": np.bitwise_xor,
}


def _propagate_const(eqn: JaxprEqn, state_consts: StateConsts) -> None:
    """Propagate a const value through a binary elementwise op.

    If both inputs are statically known,
    apply the matching host evaluator and store a non-floating result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    evaluate = _BINARY_CONST_UFUNCS.get(eqn.primitive.name)
    if evaluate is not None:
        _propagate_const_binary(eqn, state_consts, evaluate)


# Building blocks (private)


def _zero_derivative(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Set empty index sets for zero-derivative outputs."""
    for outvar in eqn.outvars:
        state_indices[outvar] = _empty_index_sets(_atom_numel(outvar))


def _binary_elementwise(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    *,
    is_der1_zero_globally: bool = False,
    is_der2_zero_globally: bool = False,
) -> None:
    """Union per-element index sets from two inputs.

    If ``is_der1_zero_globally`` is True, ∂f/∂x₁ = 0 everywhere,
    so the first input doesn't contribute to output dependencies.
    Likewise for ``is_der2_zero_globally`` and the second input.
    """
    in1 = _index_sets(state_indices, eqn.invars[0])
    in2 = _index_sets(state_indices, eqn.invars[1])
    out_size = 0 if len(in1) == 0 or len(in2) == 0 else max(len(in1), len(in2))

    in1_shape = _atom_shape(eqn.invars[0])
    in2_shape = _atom_shape(eqn.invars[1])

    # Fast path: same shape or scalar.
    # Modular indexing handles both correctly:
    # i % len == i for same size, i % 1 == 0 for scalar.
    if in1_shape == in2_shape or len(in1) <= 1 or len(in2) <= 1:
        state_indices[eqn.outvars[0]] = [
            _union_with_zero_derivs(
                in1[i % len(in1)],
                in2[i % len(in2)],
                is_der1_zero_globally,
                is_der2_zero_globally,
            )
            for i in range(out_size)
        ]
        return

    # General broadcast: out[p,d] of (16,16) * (16,1) depends on in1[p,d]
    # and in2[p,0], not in2[d].
    out_shape = _atom_shape(eqn.outvars[0])
    in1_flat = _broadcast_flat_map(in1_shape, out_shape)
    in2_flat = _broadcast_flat_map(in2_shape, out_shape)

    state_indices[eqn.outvars[0]] = [
        _union_with_zero_derivs(
            in1[in1_flat[i]],
            in2[in2_flat[i]],
            is_der1_zero_globally,
            is_der2_zero_globally,
        )
        for i in range(_numel(out_shape))
    ]


def _broadcast_flat_map(
    in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> np.ndarray:
    """Flat input position read by each output element under numpy broadcasting.

    The input shape is left-padded with 1s to the output rank;
    size-1 dimensions always read index 0.
    """
    ndim = len(out_shape)
    padded = (1,) * (ndim - len(in_shape)) + in_shape
    out_coords = np.indices(out_shape)
    coords = tuple(out_coords[d] if padded[d] > 1 else 0 for d in range(ndim))
    return np.broadcast_to(np.ravel_multi_index(coords, padded), out_shape).ravel()


def _union_with_zero_derivs(
    s1: set[int],
    s2: set[int],
    is_der1_zero: bool,
    is_der2_zero: bool,
) -> set[int]:
    """Union index sets, excluding inputs with zero derivatives."""
    if is_der1_zero and is_der2_zero:
        return set()
    if is_der1_zero:
        return s2.copy()
    if is_der2_zero:
        return s1.copy()
    return s1 | s2


def _propagate_bounds_add(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate value bounds through ``add`` or ``add_any`` via interval arithmetic."""
    b1 = _atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    b2 = _atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if b1 is None or b2 is None:
        return
    lo1, hi1 = b1
    lo2, hi2 = b2
    _store_bounds(eqn, state_bounds, lo1 + lo2, hi1 + hi2)


def _propagate_bounds_sub(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate value bounds through ``sub`` via interval arithmetic."""
    b1 = _atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    b2 = _atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if b1 is None or b2 is None:
        return
    lo1, hi1 = b1
    lo2, hi2 = b2
    _store_bounds(eqn, state_bounds, lo1 - hi2, hi1 - lo2)


def _propagate_bounds_min_max(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate value bounds through ``min`` or ``max``.

    Both are monotone nondecreasing in each operand, so the result interval is
    the same extremum of the operand endpoints. An unbounded operand leaves the
    result unbounded: one known side is not a bound on the other.
    """
    b1 = _atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    b2 = _atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if b1 is None or b2 is None:
        return
    extremum = np.minimum if eqn.primitive.name == "min" else np.maximum
    _store_bounds(eqn, state_bounds, extremum(b1[0], b2[0]), extremum(b1[1], b2[1]))


# Composite handlers (public)
# Each corresponds to exactly one dispatch case in _prop_dispatch.


def _prop_zero_derivative(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Zero-derivative primitives (floor, ceil, sign, ...).

    Operations with zero derivative almost everywhere.
    Their outputs are piecewise constant,
    so infinitesimal input changes don't affect outputs.

    Mathematically, for f in {floor, ceil, sign, ...}:
        ∂f/∂x = 0  (almost everywhere)
    Therefore, output elements have no dependencies on input elements.

    Example: y = floor(x) where x = [1.7, 2.3, 3.9]
        Input state_indices:  [{0}, {1}, {2}]
        Output state_indices: [{}, {}, {}]  (empty sets, no dependence)
    """
    _zero_derivative(eqn, state_indices)


def _prop_zero_derivative_const(
    eqn: JaxprEqn, state_indices: StateIndices, state_consts: StateConsts
) -> None:
    """Zero-derivative primitives that also propagate const values.

    Used for comparisons (eq, ne) and bitwise ops (and, or, xor)
    where the output is zero-derivative
    but the concrete result may be needed by downstream handlers.
    """
    _zero_derivative(eqn, state_indices)
    _propagate_const(eqn, state_consts)


def _prop_ternary_elementwise(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Ternary elementwise operation where each output depends on all three inputs.

    Used for `regularized_incomplete_beta(a, b, x)` where each output element
    depends on the corresponding elements from all three input arrays.
    Handles broadcasting via modular indexing.

    Example: z = betainc(a, b, x) where a, b, x are arrays of shape (3,)
        Input state_indices:  [{0}, {1}, {2}], [{3}, {4}, {5}], [{6}, {7}, {8}]
        Output state_indices: [{0, 3, 6}, {1, 4, 7}, {2, 5, 8}]

    Jaxpr:
        invars[0]: first input (a)
        invars[1]: second input (b)
        invars[2]: third input (x)
    """
    inputs = [_index_sets(state_indices, invar) for invar in eqn.invars]
    out_size = _atom_numel(eqn.outvars[0])
    state_indices[eqn.outvars[0]] = _union_elementwise(inputs, out_size)


def _prop_binary_const(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    *,
    is_der1_zero_globally: bool = False,
    is_der2_zero_globally: bool = False,
) -> None:
    """Binary elementwise primitives (pow, atan2, rem, ...) with const propagation.

    Each output element depends on the corresponding elements from both inputs.
    Also propagates const values for downstream index resolution.

    For f(x, y) element-wise:
        ∂f/∂x[i] and ∂f/∂y[i] are generally nonzero.
    So out[i] depends on {x[i], y[i]} (union of dependencies).

    If ``is_der1_zero_globally`` is True, ∂f/∂x₁ = 0 everywhere,
    so only the second input contributes.
    Likewise for ``is_der2_zero_globally``.

    Example: z = x + y where x = [a, b], y = [c, d]
        Input state_indices:  [{0}, {1}], [{2}, {3}]
        Output state_indices: [{0, 2}, {1, 3}]

    Jaxpr:
        invars[0]: first input array
        invars[1]: second input array
    """
    _binary_elementwise(
        eqn,
        state_indices,
        is_der1_zero_globally=is_der1_zero_globally,
        is_der2_zero_globally=is_der2_zero_globally,
    )
    _propagate_const(eqn, state_consts)


def _prop_add(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Add / add_any: binary elementwise with interval arithmetic bounds.

    ``[a,b] + [c,d] = [a+c, b+d]``.
    """
    _binary_elementwise(eqn, state_indices)
    _propagate_const(eqn, state_consts)
    _propagate_bounds_add(eqn, state_consts, state_bounds)


def _prop_sub(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Sub: binary elementwise with interval arithmetic bounds.

    ``[a,b] - [c,d] = [a-d, b-c]``.
    """
    _binary_elementwise(eqn, state_indices)
    _propagate_const(eqn, state_consts)
    _propagate_bounds_sub(eqn, state_consts, state_bounds)


def _prop_min_max(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Min / max: binary elementwise with monotone interval bounds.

    ``∂min/∂x = 1`` where ``x < y`` and ``∂min/∂y = 1`` where ``y < x``, so
    each output depends on both operands. Value bounds follow
    ``min([a,b], [c,d]) = [min(a,c), min(b,d)]`` and likewise for ``max``,
    which keeps clamped index arithmetic bounded for gather and scatter.
    """
    _binary_elementwise(eqn, state_indices)
    _propagate_const(eqn, state_consts)
    _propagate_bounds_min_max(eqn, state_consts, state_bounds)


def _prop_integer_pow(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts | None = None,
    state_bounds: StateBounds | None = None,
) -> None:
    """Integer power x^n is element-wise.

    Each output depends only on the corresponding input element.
    Special cases:
    - x^0 = 1 has zero derivative, so no dependencies.
    - 0^n = 0 for n > 0, so d(0^n)/dx = 0 and no dependencies.

    ∂(x^n)/∂x = n·x^(n-1), which is zero iff n = 0 or (x = 0 and n > 1).

    Example: y = x^2 where x = [a, b, c]
        Input state_indices:  [{0}, {1}, {2}]
        Output state_indices: [{0}, {1}, {2}]  (or [{}, {}, {}] if n=0)

    Jaxpr:
        invars[0]: input array
        y: the integer exponent

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.integer_pow.html
    """
    y = eqn.params.get("y", 1)
    in_indices = _index_sets(state_indices, eqn.invars[0])

    if y == 0:
        state_indices[eqn.outvars[0]] = _empty_index_sets(len(in_indices))
    else:
        state_indices[eqn.outvars[0]] = _copy_index_sets(in_indices)

    # Const propagation.
    if state_consts is not None:
        in_val = _atom_const_val(eqn.invars[0], state_consts)
        if in_val is not None:
            _store_folded_const(state_consts, eqn.outvars[0], np.power(in_val, y))

    # Zero-skipping: d(0^n)/dx = n * 0^(n-1) = 0 for n > 1.
    # For n = 1, d(x)/dx = 1 even at x = 0, so no skipping.
    if state_consts is not None and y > 1:
        _clear_where_zero(eqn, state_indices, state_consts, 0)

    # Bounds propagation for [a,b]^n.
    if state_bounds is not None:
        _propagate_bounds_integer_pow(eqn, y, state_consts or {}, state_bounds)


def _propagate_bounds_integer_pow(
    eqn: JaxprEqn,
    y: int,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate integer value bounds through ``integer_pow``.

    - n == 0: bounds are (1, 1).
    - n even: [0, max(|a|,|b|)^n] if interval spans zero,
      else [min(|a|,|b|)^n, max(|a|,|b|)^n].
    - n odd (monotone): [a^n, b^n].

    Floating powers are evaluated by backend-specific multiplication chains, so
    their host values are not bounds and none are recorded.
    """
    in_bounds = _atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    if in_bounds is None:
        return

    lo, hi = in_bounds
    if _is_floating(lo) or _is_floating(hi):
        return

    if y == 0:
        ones = np.ones_like(lo)
        state_bounds[eqn.outvars[0]] = (ones, ones)
    elif y % 2 == 1:
        # Odd power is monotone.
        state_bounds[eqn.outvars[0]] = (np.power(lo, y), np.power(hi, y))
    else:
        # Even power: x^n is not monotone over intervals spanning zero.
        abs_lo = np.abs(lo)
        abs_hi = np.abs(hi)
        max_abs = np.maximum(abs_lo, abs_hi)
        min_abs = np.minimum(abs_lo, abs_hi)

        spans_zero = (lo <= 0) & (hi >= 0)
        out_lo = np.where(spans_zero, np.zeros_like(lo), np.power(min_abs, y))
        out_hi = np.power(max_abs, y)
        state_bounds[eqn.outvars[0]] = (out_lo, out_hi)


def _prop_unary_elementwise(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Unary element-wise ops (exp, sin, etc.) apply a function to each element.

    Each output depends only on the corresponding input element.
    The Jacobian is diagonal.

    For f(x) element-wise:
        ∂f[i]/∂x[j] = f'(x[i]) if i = j, else 0

    Example: y = exp(x) where x = [a, b, c]
        Input state_indices:  [{0}, {1}, {2}]
        Output state_indices: [{0}, {1}, {2}]

    Jaxpr:
        invars[0]: input array
    """
    state_indices[eqn.outvars[0]] = _copy_index_sets(
        _index_sets(state_indices, eqn.invars[0])
    )


def _prop_sign(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Sign is piecewise constant on reals but not on complex numbers.

    Real ``sign(x)`` has ``∂/∂x = 0`` almost everywhere.
    Complex ``sign(z) = z / |z|`` varies along the unit circle,
    so ``sign(z)[i]`` depends on ``z[i]``.
    """
    aval = eqn.invars[0].aval
    if not isinstance(aval, ShapedArray):
        raise TypeError("Structural sign propagation requires a shaped array.")
    if np.issubdtype(aval.dtype, np.complexfloating):
        _prop_unary_elementwise(eqn, state_indices)
    else:
        _zero_derivative(eqn, state_indices)


def _prop_polynomial(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Polynomial ``a₀ + x·(a₁ + x·(…))`` over broadcast operands.

    Each output element depends on its broadcast element of ``x`` and of
    every coefficient. No entry is pruned at sampled zeros:
    ``∂p/∂a₀ = 1`` and the other partials vanish only at isolated points.

    Jaxpr:
        invars[0]: argument x
        invars[1:]: coefficients (a₀, …, a_N), lowest degree first
    """
    out_shape = _atom_shape(eqn.outvars[0])
    out_indices = _empty_index_sets(_numel(out_shape))
    for invar in eqn.invars:
        in_indices = _index_sets(state_indices, invar)
        flat = _broadcast_flat_map(_atom_shape(invar), out_shape)
        for i, j in enumerate(flat):
            out_indices[i] |= in_indices[j]
    state_indices[eqn.outvars[0]] = out_indices


def _prop_convert_element_type(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds | None = None,
) -> None:
    """Type conversion (e.g., float32 → float64) changes dtype without changing values.

    Dependencies pass through unchanged.
    The Jacobian is the identity matrix.

    Also propagates const values and value bounds with the new dtype
    so downstream gather/scatter can resolve static indices.
    JAX inserts ``convert_element_type`` for index dtype changes
    (e.g. int64 → int32) before gather/scatter;
    without const propagation here the chain breaks
    and gathers fall back to conservative.

    Example: y = x.astype(float64) where x = [a, b, c]
        Input state_indices:  [{0}, {1}, {2}]
        Output state_indices: [{0}, {1}, {2}]

    Jaxpr:
        invars[0]: input array
        new_dtype: target dtype

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html
    """
    state_indices[eqn.outvars[0]] = _copy_index_sets(
        _index_sets(state_indices, eqn.invars[0])
    )

    in_val = _atom_const_val(eqn.invars[0], state_consts)
    if in_val is not None:
        converted = _converted_const(eqn, in_val)
        if converted is not None:
            state_consts[eqn.outvars[0]] = converted

    # Propagate value bounds through value conversions.
    if state_bounds is not None and eqn.primitive.name in (
        "convert_element_type",
        "stop_gradient",
    ):
        bounds = _atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
        if bounds is not None:
            new_dtype = eqn.params.get("new_dtype")
            converted = (
                bounds
                if new_dtype is None
                else _converted_bounds(eqn, *bounds, np.dtype(new_dtype))
            )
            if converted is not None:
                state_bounds[eqn.outvars[0]] = converted


def _converted_bounds(
    eqn: JaxprEqn, lower: np.ndarray, upper: np.ndarray, new_dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounds after ``convert_element_type``; conversions are monotone.

    Floating targets round, so they are widened like arithmetic results.
    Floating-to-integer truncation is bounded for finite in-range endpoints;
    integer narrowing is bounded only when both endpoints are preserved.
    """
    if any(jnp.issubdtype(end.dtype, jnp.complexfloating) for end in (lower, upper)):
        return None
    if jnp.issubdtype(new_dtype, jnp.inexact):
        return _backend_bounds(eqn, lower, upper)
    if not jnp.issubdtype(new_dtype, jnp.integer):
        return None
    if _is_floating(lower) or _is_floating(upper):
        low, high = np.trunc(lower), np.trunc(upper)
        if not (
            _truncates_in_range(low, new_dtype) and _truncates_in_range(high, new_dtype)
        ):
            return None
        return low.astype(new_dtype), high.astype(new_dtype)
    if all(
        np.array_equal(end.astype(new_dtype).astype(end.dtype), end)
        for end in (lower, upper)
    ):
        return lower.astype(new_dtype), upper.astype(new_dtype)
    return None


def _converted_const(eqn: JaxprEqn, value: np.ndarray) -> np.ndarray | None:
    """Host value of a conversion that every backend computes identically.

    ``stop_gradient`` keeps the value and ``bitcast_convert_type`` reinterprets
    its bits. ``reduce_precision`` rounds floating data, so it is not folded.
    Floating-to-integer conversion truncates toward zero and is folded only for
    finite values in the target range, where XLA and NumPy agree.
    """
    match eqn.primitive.name:
        case "stop_gradient":
            return value
        case "bitcast_convert_type":
            new_dtype = np.dtype(eqn.params["new_dtype"])
            if new_dtype.itemsize != value.dtype.itemsize:
                return None
            return np.ascontiguousarray(value).view(new_dtype)
        case "convert_element_type":
            new_dtype = np.dtype(eqn.params["new_dtype"])
            if (
                jnp.issubdtype(new_dtype, jnp.integer)
                and _is_floating(value)
                and not _truncates_in_range(np.trunc(np.real(value)), new_dtype)
            ):
                return None
            return value.astype(new_dtype)
        case _:
            return None


def _truncates_in_range(truncated: np.ndarray, new_dtype: np.dtype) -> bool:
    """Whether truncated floating values are finite and representable in ``new_dtype``.

    XLA and NumPy disagree on non-finite and out-of-range conversions.
    """
    limits = jnp.iinfo(new_dtype)
    return bool(
        np.all(
            np.isfinite(truncated)
            & (truncated >= float(limits.min))
            & (truncated < float(limits.max) + 1.0)
        )
    )


def _prop_clamp(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Clamp(lo, x, hi) returns lo when x < lo, hi when x > hi, else x.

    All three operands can contribute to the output depending on runtime values:
        ∂clamp/∂lo = 1 if x < lo, else 0
        ∂clamp/∂x  = 1 if lo ≤ x ≤ hi, else 0
        ∂clamp/∂hi = 1 if x > hi, else 0

    Since we compute global sparsity patterns (not runtime-dependent),
    we conservatively propagate dependencies from all three operands.

    Example: y = clamp(x[0], x[1], x[2]) where x = [lo, val, hi]
        Input state_indices: lo=[{0}], val=[{1}], hi=[{2}]
        Output state_indices: [{0, 1, 2}]

    Jaxpr:
        invars[0]: lo (lower bound)
        invars[1]: x (value to clamp)
        invars[2]: hi (upper bound)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.clamp.html
    """
    lo = _index_sets(state_indices, eqn.invars[0])
    x = _index_sets(state_indices, eqn.invars[1])
    hi = _index_sets(state_indices, eqn.invars[2])

    state_indices[eqn.outvars[0]] = _union_elementwise(
        [lo, x, hi], _atom_numel(eqn.outvars[0])
    )

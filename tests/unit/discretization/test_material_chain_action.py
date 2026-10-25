#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Behavior of the factorized chronological material-chain action.

Schedules are built directly from token-slot chunks; every result is checked
against an exact rational (``Fraction``) or dense differentiable oracle of the
same chronological product.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.discretization.finite_volume._material_chain_action import (
    compose,
    material_chain_apply,
    material_chain_transpose,
    split,
    ZERO_EXPONENT,
)
from phydrax.discretization.finite_volume._material_schedule import MaterialChainSchedule
from phydrax.sparse import InjectiveRouteWindows


pytestmark = [
    pytest.mark.strict_jax,
    pytest.mark.skipif(
        not bool(pxconfig.x64_enabled()),
        reason="The material-chain action is checked in float64 and requires JAX x64.",
    ),
]

_REPLAY = (1 << 30, 1 << 30)
_FLOATS = (np.float64, np.float32)
_UNSIGNED = {np.float64: np.uint64, np.float32: np.uint32}


# ---------------------------------------------------------------------------
# IEEE split / compose.


def _bits(values: Any, dtype: Any) -> np.ndarray:
    return np.asarray(values, dtype=dtype).view(_UNSIGNED[dtype])


def _random_bit_patterns(dtype: Any, count: int, seed: int) -> np.ndarray:
    unsigned = _UNSIGNED[dtype]
    rng = np.random.default_rng(seed)
    return rng.integers(
        0, np.iinfo(unsigned).max, size=count, dtype=unsigned, endpoint=True
    ).view(dtype)


def _boundary_values(dtype: Any) -> np.ndarray:
    info = np.finfo(dtype)
    tiny = info.smallest_normal
    return np.asarray(
        [
            info.smallest_subnormal,
            -info.smallest_subnormal,
            np.nextafter(tiny, dtype(0)),
            tiny,
            1.0,
            -1.0,
            info.max,
            -info.max,
            0.1,
            -3.5,
        ],
        dtype=dtype,
    )


def _host_ldexp(
    mantissa: float | np.floating, exponent: int | np.integer, dtype: Any
) -> np.ndarray:
    """Correctly rounded (nearest, ties to even) ``mantissa * 2**exponent`` in ``dtype``.

    Exact rational rounding: the host ``math.ldexp`` is not a reliable oracle
    in the subnormal range (it can truncate instead of rounding).
    """
    info = np.finfo(dtype)
    value = Fraction(float(mantissa)) * Fraction(2) ** int(exponent)
    sign = -1.0 if math.copysign(1.0, float(mantissa)) < 0 else 1.0
    magnitude = abs(value)
    if magnitude == 0:
        return np.asarray(sign * 0.0, dtype=dtype)
    binade = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < Fraction(2) ** binade:
        binade -= 1
    unit = Fraction(2) ** (max(binade, int(info.minexp)) - int(info.nmant))
    quotient = magnitude / unit
    whole = quotient.numerator // quotient.denominator
    remainder = quotient - whole
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and whole % 2):
        whole += 1
    rounded = whole * unit
    if rounded > Fraction(float(info.max)):
        return np.asarray(sign * math.inf, dtype=dtype)
    return np.asarray(sign * float(rounded), dtype=dtype)


@pytest.mark.parametrize("dtype", _FLOATS)
def test_split_matches_host_frexp_on_finite_values(dtype: Any) -> None:
    patterns = _random_bit_patterns(dtype, 512, seed=3)
    random = patterns[np.isfinite(patterns) & (patterns != 0)]
    normals = (np.random.default_rng(4).standard_normal(64) * 1.0e3).astype(dtype)
    values = np.concatenate([_boundary_values(dtype), random, normals])

    mantissa, exponent = jax.jit(split)(jnp.asarray(values))
    mantissa = np.asarray(mantissa)
    exponent = np.asarray(exponent)

    assert mantissa.dtype == dtype
    assert exponent.dtype == np.int32
    assert np.all((np.abs(mantissa) >= 0.5) & (np.abs(mantissa) < 1.0))
    for value, m, e in zip(values, mantissa, exponent, strict=True):
        assert (float(m), int(e)) == math.frexp(float(value)), value


@pytest.mark.parametrize("dtype", _FLOATS)
def test_split_exact_pairs_at_the_subnormal_and_overflow_boundaries(dtype: Any) -> None:
    info = np.finfo(dtype)
    fraction_bits = int(info.nmant)
    lowest = int(info.minexp)  # exponent of the least normal, 2**lowest
    values = np.asarray(
        [
            info.smallest_subnormal,
            np.nextafter(info.smallest_normal, dtype(0)),
            info.smallest_normal,
            1.0,
            info.max,
        ],
        dtype=dtype,
    )
    expected = [
        (0.5, lowest - fraction_bits + 1),
        (1.0 - float(info.eps), lowest),
        (0.5, lowest + 1),
        (0.5, 1),
        (float(np.nextafter(dtype(1), dtype(0))), int(info.maxexp)),
    ]

    mantissa, exponent = split(jnp.asarray(values))

    assert [
        (float(m), int(e)) for m, e in zip(np.asarray(mantissa), np.asarray(exponent), strict=True)
    ] == expected


@pytest.mark.parametrize("dtype", _FLOATS)
def test_split_signed_zero_and_nonfinite_values(dtype: Any) -> None:
    values = np.asarray([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=dtype)

    mantissa, exponent = split(jnp.asarray(values))
    mantissa = np.asarray(mantissa)
    exponent = np.asarray(exponent)

    np.testing.assert_array_equal(_bits(mantissa[:4], dtype), _bits(values[:4], dtype))
    assert np.isnan(mantissa[4])
    np.testing.assert_array_equal(exponent, [ZERO_EXPONENT, ZERO_EXPONENT, 0, 0, 0])


@pytest.mark.parametrize("dtype", _FLOATS)
def test_split_then_compose_round_trips_every_bit_pattern(dtype: Any) -> None:
    values = np.concatenate(
        [
            _boundary_values(dtype),
            np.asarray([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=dtype),
            _random_bit_patterns(dtype, 2048, seed=5),
        ]
    )

    composed = np.asarray(jax.jit(lambda v: compose(*split(v)))(jnp.asarray(values)))

    nan = np.isnan(values)
    np.testing.assert_array_equal(np.isnan(composed), nan)
    np.testing.assert_array_equal(_bits(composed[~nan], dtype), _bits(values[~nan], dtype))


@pytest.mark.parametrize("dtype", _FLOATS)
def test_compose_matches_host_ldexp(dtype: Any) -> None:
    info = np.finfo(dtype)
    rng = np.random.default_rng(6)
    count = 4096
    # Clamp below one: rounding a uniform draw to float32 may reach 1.0.
    magnitudes = np.minimum(
        rng.uniform(0.5, 1.0, size=count).astype(dtype), np.nextafter(dtype(1), dtype(0))
    )
    mantissas = (magnitudes * rng.choice([-1.0, 1.0], size=count).astype(dtype)).astype(dtype)
    mantissas[0] = dtype(0.5)
    mantissas[1] = -np.nextafter(dtype(1), dtype(0))
    low = int(info.minexp) - int(info.nmant) - 8
    high = int(info.maxexp) + 4
    exponents = rng.integers(low, high, size=count, endpoint=True).astype(np.int32)

    composed = np.asarray(
        jax.jit(compose)(jnp.asarray(mantissas), jnp.asarray(exponents))
    )

    expected = np.asarray(
        [_host_ldexp(m, int(e), dtype) for m, e in zip(mantissas, exponents, strict=True)],
        dtype=dtype,
    )
    np.testing.assert_array_equal(_bits(composed, dtype), _bits(expected, dtype))


@pytest.mark.parametrize("dtype", _FLOATS)
def test_compose_rounds_to_nearest_even_into_the_subnormal_range(dtype: Any) -> None:
    info = np.finfo(dtype)
    sub = int(info.minexp) - int(info.nmant)  # least subnormal is 2**sub
    almost_one = float(np.nextafter(dtype(1), dtype(0)))
    cases = [
        (0.75, sub + 1, 2.0 ** (sub + 1)),  # 1.5 ulp: tie, rounds up to even 2
        (0.625, sub + 2, 2.0 ** (sub + 1)),  # 2.5 ulp: tie, rounds down to even 2
        (0.5, sub, 0.0),  # 0.5 ulp: tie, rounds down to even 0
        (-0.5, sub, -0.0),  # signed tie to zero keeps the sign
        (0.75, sub, 2.0**sub),  # 0.75 ulp: rounds up to the least subnormal
        (0.5, sub + 1, 2.0**sub),  # exact least subnormal
        (-0.75, sub + 1, -(2.0 ** (sub + 1))),
        (almost_one, int(info.minexp), 2.0 ** int(info.minexp)),  # carry to least normal
    ]
    mantissas = np.asarray([case[0] for case in cases], dtype=dtype)
    exponents = np.asarray([case[1] for case in cases], dtype=np.int32)
    expected = np.asarray([case[2] for case in cases], dtype=dtype)

    composed = np.asarray(compose(jnp.asarray(mantissas), jnp.asarray(exponents)))

    np.testing.assert_array_equal(_bits(composed, dtype), _bits(expected, dtype))
    host = np.asarray(
        [_host_ldexp(m, e, dtype) for m, e in zip(mantissas, exponents, strict=True)],
        dtype=dtype,
    )
    np.testing.assert_array_equal(_bits(composed, dtype), _bits(host, dtype))


@pytest.mark.parametrize("dtype", _FLOATS)
def test_compose_overflow_threshold(dtype: Any) -> None:
    info = np.finfo(dtype)
    top = int(info.maxexp)
    almost_one = float(np.nextafter(dtype(1), dtype(0)))
    cases = [
        (almost_one, top, info.max),  # largest finite stays finite
        (-almost_one, top, -info.max),
        (0.5, top, 2.0 ** (top - 1)),
        (0.5, top + 1, np.inf),  # next binade overflows
        (-0.5, top + 1, -np.inf),
        (0.5, 1 << 20, np.inf),
        (0.5, ZERO_EXPONENT, 0.0),
        (-0.5, ZERO_EXPONENT, -0.0),
    ]
    mantissas = np.asarray([case[0] for case in cases], dtype=dtype)
    exponents = np.asarray([case[1] for case in cases], dtype=np.int32)
    expected = np.asarray([case[2] for case in cases], dtype=dtype)

    composed = np.asarray(compose(jnp.asarray(mantissas), jnp.asarray(exponents)))

    np.testing.assert_array_equal(_bits(composed, dtype), _bits(expected, dtype))


# ---------------------------------------------------------------------------
# Hand-built schedules and oracles.


@dataclass(frozen=True)
class _Chunk:
    reads: tuple[int, ...]
    writes: tuple[int, ...]
    # (local read lane, local write lane, global weight row, exact exponent)
    lanes: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class _Spec:
    sources: int
    slots: int
    finals: tuple[int, ...]
    weights: int
    chunks: tuple[_Chunk, ...]

    @property
    def targets(self) -> int:
        return len(self.finals)


# One chunk: both sources read, slots swapped and mixed.
_SINGLE = _Spec(
    sources=2,
    slots=2,
    finals=(0, 1),
    weights=3,
    chunks=(_Chunk((0, 1), (1, 0), ((0, 0, 0, 2), (1, 0, 1, -1), (1, 1, 2, 0))),),
)

# Three chunks: slot 0 is recreated in chunk 0 and then passes through
# untouched; chunk 1 reuses slots 1 and 2 that it reads; chunk 2 rescales.
_THREE = _Spec(
    sources=2,
    slots=4,
    finals=(0, 1, 3, 2),
    weights=7,
    chunks=(
        _Chunk((0,), (0, 2), ((0, 0, 0, 1), (0, 1, 1, -2))),
        _Chunk(
            (2, 1),
            (1, 3, 2),
            ((0, 0, 2, 3), (1, 0, 3, 0), (1, 1, 4, -1), (0, 2, 5, 2)),
        ),
        _Chunk((3,), (3,), ((0, 0, 6, -3),)),
    ),
)

# Merge then split: slot 0 ends holding a consumed token that is not a target.
_DEAD = _Spec(
    sources=2,
    slots=3,
    finals=(1, 2),
    weights=4,
    chunks=(
        _Chunk((0, 1), (0,), ((0, 0, 0, -1), (1, 0, 1, 1))),
        _Chunk((0,), (2, 1), ((0, 0, 2, 0), (0, 1, 3, -2))),
    ),
)


def _long_spec(chunks: int, /) -> _Spec:
    """Two live tokens over three slots; each chunk mixes both into two new tokens."""
    live = [0, 1]
    free = 2
    row = 0
    built = []
    for index in range(chunks):
        first, second = live
        built.append(
            _Chunk(
                (first, second),
                (free, first),
                (
                    (0, 0, row, index % 3 - 1),
                    (1, 0, row + 1, 1 - 2 * (index % 2)),
                    (1, 1, row + 2, index % 4 - 2),
                ),
            )
        )
        live, free, row = [free, first], second, row + 3
    return _Spec(sources=2, slots=3, finals=tuple(live), weights=row, chunks=tuple(built))


_LONG = _long_spec(6)

# Compress then expand by 2**-1000 / 2**1000: every end-to-end weight is O(1).
_RANGE = _Spec(
    sources=2,
    slots=2,
    finals=(0, 1),
    weights=6,
    chunks=(
        _Chunk((0, 1), (0, 1), ((0, 0, 0, -1000), (1, 1, 1, 1000), (1, 0, 2, -1000))),
        _Chunk((0, 1), (1, 0), ((0, 0, 3, 1000), (1, 1, 4, -1000), (0, 1, 5, 1000))),
    ),
)

_SPECS = {"single": _SINGLE, "three": _THREE, "dead": _DEAD, "long": _LONG}
_MODES = ("plain", "denominators", "source_measures")


def _windows(rows: Sequence[Sequence[int]], size: int, /) -> InjectiveRouteWindows:
    capacity = max(len(row) for row in rows)
    indices = np.full((len(rows), capacity), size, dtype=np.int64)
    valid = np.zeros((len(rows), capacity), dtype=np.bool_)
    for window, row in enumerate(rows):
        indices[window, : len(row)] = row
        valid[window, : len(row)] = True
    return InjectiveRouteWindows(indices, target_size=size, valid=valid)


def _schedule(spec: _Spec, name: str, /) -> MaterialChainSchedule:
    capacity = max(len(chunk.lanes) for chunk in spec.chunks)
    lanes = np.zeros((3, len(spec.chunks), capacity), dtype=np.int32)
    for index, chunk in enumerate(spec.chunks):
        for lane, (read, write, _, exponent) in enumerate(chunk.lanes):
            lanes[:, index, lane] = (read, write, exponent)
    return MaterialChainSchedule(
        _windows([chunk.reads for chunk in spec.chunks], spec.slots),
        _windows([chunk.writes for chunk in spec.chunks], spec.slots),
        _windows([spec.finals], spec.slots),
        _windows(
            [tuple(lane[2] for lane in chunk.lanes) for chunk in spec.chunks], spec.weights
        ),
        jnp.asarray(lanes[0]),
        jnp.asarray(lanes[1]),
        jnp.asarray(lanes[2]),
        source_count=spec.sources,
        target_count=spec.targets,
    )


def _exact_operator(spec: _Spec, mantissas: np.ndarray, /) -> list[list[Fraction]]:
    """Exact density map ``A`` as rows over the sources, untouched slots passing through."""
    count = spec.sources
    state = {
        slot: [Fraction(int(slot == source)) for source in range(count)]
        for slot in range(spec.slots)
    }
    for chunk in spec.chunks:
        old = [state[slot] for slot in chunk.reads]
        new = [[Fraction(0)] * count for _ in chunk.writes]
        for read, write, row, exponent in chunk.lanes:
            weight = Fraction(float(mantissas[row])) * Fraction(2) ** exponent
            new[write] = [a + weight * b for a, b in zip(new[write], old[read], strict=True)]
        state.update(zip(chunk.writes, new, strict=True))
    return [state[slot] for slot in spec.finals]


def _fractions(values: Any, /) -> list[Fraction]:
    return [Fraction(float(value)) for value in np.asarray(values, dtype=np.float64)]


def _exact_on_parts(values: np.ndarray, rows: int, real: Any, /) -> np.ndarray:
    """Apply the real exact map ``real`` to real and imaginary parts of ``values``."""
    payload = values.shape[1:]
    flat = values.reshape((values.shape[0], -1))
    if np.iscomplexobj(values):
        result = np.empty((rows, flat.shape[1]), dtype=np.complex128)
        result.real = real(flat.real)
        result.imag = real(flat.imag)
    else:
        result = real(flat)
    return result.reshape((rows, *payload))


def _exact_apply(
    spec: _Spec,
    mantissas: Any,
    measures: Any,
    values: Any,
    /,
    *,
    denominators: Any = None,
    source_measures: Any = None,
) -> np.ndarray:
    operator = _exact_operator(spec, np.asarray(mantissas))
    factor = _fractions(measures)
    if denominators is not None:
        factor = [m / d for m, d in zip(factor, _fractions(denominators), strict=True)]
    scale = (
        [Fraction(1)] * spec.sources
        if source_measures is None
        else [1 / s for s in _fractions(source_measures)]
    )

    def real(part: np.ndarray) -> np.ndarray:
        out = np.empty((spec.targets, part.shape[1]), dtype=np.float64)
        for target in range(spec.targets):
            for channel in range(part.shape[1]):
                column = _fractions(part[:, channel])
                total = sum(
                    (a * s * x for a, s, x in zip(operator[target], scale, column, strict=True)),
                    Fraction(0),
                )
                out[target, channel] = float(factor[target] * total)
        return out

    return _exact_on_parts(np.asarray(values), spec.targets, real)


def _exact_transpose(
    spec: _Spec,
    mantissas: Any,
    measures: Any,
    cotangents: Any,
    /,
    *,
    denominators: Any = None,
) -> np.ndarray:
    operator = _exact_operator(spec, np.asarray(mantissas))
    factor = _fractions(measures)
    if denominators is not None:
        factor = [m / d for m, d in zip(factor, _fractions(denominators), strict=True)]

    def real(part: np.ndarray) -> np.ndarray:
        out = np.empty((spec.sources, part.shape[1]), dtype=np.float64)
        for source in range(spec.sources):
            for channel in range(part.shape[1]):
                column = _fractions(part[:, channel])
                total = sum(
                    (
                        operator[target][source] * factor[target] * column[target]
                        for target in range(spec.targets)
                    ),
                    Fraction(0),
                )
                out[source, channel] = float(total)
        return out

    return _exact_on_parts(np.asarray(cotangents), spec.sources, real)


def _naive_apply(
    spec: _Spec,
    mantissas: Any,
    measures: Any,
    values: Any,
    /,
    *,
    denominators: Any = None,
    source_measures: Any = None,
) -> np.ndarray:
    """Chunk-by-chunk float64 densities, the computation the pair arithmetic replaces."""
    mantissas = np.asarray(mantissas)
    values = np.asarray(values)
    flat = values.reshape((spec.sources, -1))
    with np.errstate(all="ignore"):
        if source_measures is not None:
            flat = flat / np.asarray(source_measures)[:, None]
        state = np.zeros((spec.slots, flat.shape[1]), dtype=flat.dtype)
        state[: spec.sources] = flat
        for chunk in spec.chunks:
            old = state[list(chunk.reads)]
            new = np.zeros((len(chunk.writes), flat.shape[1]), dtype=flat.dtype)
            for read, write, row, exponent in chunk.lanes:
                new[write] += mantissas[row] * 2.0**exponent * old[read]
            state[list(chunk.writes)] = new
        factor = np.asarray(measures)
        if denominators is not None:
            factor = factor / np.asarray(denominators)
        out = factor[:, None] * state[list(spec.finals)]
    return out.reshape((spec.targets, *values.shape[1:]))


def _oracle_operator(spec: _Spec, mantissas: jax.Array, /) -> jax.Array:
    """Dense float64 ``A(M)`` as a differentiable function of the mantissas."""
    identity = jnp.eye(spec.sources, dtype=jnp.float64)
    zero = jnp.zeros((spec.sources,), dtype=jnp.float64)
    state = {
        slot: identity[slot] if slot < spec.sources else zero for slot in range(spec.slots)
    }
    for chunk in spec.chunks:
        old = [state[slot] for slot in chunk.reads]
        new = [zero for _ in chunk.writes]
        for read, write, row, exponent in chunk.lanes:
            new[write] = new[write] + mantissas[row] * math.ldexp(1.0, exponent) * old[read]
        state.update(zip(chunk.writes, new, strict=True))
    return jnp.stack([state[slot] for slot in spec.finals])


def _endpoint(mode: str, extra: Sequence[Any], /) -> dict[str, Any]:
    if mode == "denominators":
        return {"denominators": extra[0]}
    if mode == "source_measures":
        return {"source_measures": extra[0]}
    return {}


def _kernel(
    schedule: MaterialChainSchedule, mode: str, replay: tuple[int, int] = _REPLAY
) -> Callable[..., jax.Array]:
    def apply(
        values: jax.Array, mantissas: jax.Array, measures: jax.Array, *extra: jax.Array
    ) -> jax.Array:
        return material_chain_apply(
            schedule, mantissas, measures, values, replay=replay, **_endpoint(mode, extra)
        )

    return apply


def _oracle(spec: _Spec, mode: str) -> Callable[..., jax.Array]:
    def apply(
        values: jax.Array, mantissas: jax.Array, measures: jax.Array, *extra: jax.Array
    ) -> jax.Array:
        operator = _oracle_operator(spec, mantissas)
        if mode == "source_measures":
            values = values / extra[0][:, None]
        factor = measures / extra[0] if mode == "denominators" else measures
        return factor[:, None] * (operator @ values)

    return apply


@dataclass(frozen=True)
class _Data:
    mantissas: np.ndarray
    measures: np.ndarray
    denominators: np.ndarray
    source_measures: np.ndarray

    def endpoint(self, mode: str, /) -> dict[str, Any]:
        return _endpoint(mode, self.extra(mode))

    def extra(self, mode: str, /) -> tuple[np.ndarray, ...]:
        if mode == "denominators":
            return (self.denominators,)
        if mode == "source_measures":
            return (self.source_measures,)
        return ()


def _data(spec: _Spec, seed: int, /) -> _Data:
    rng = np.random.default_rng(seed)
    return _Data(
        rng.uniform(0.5, 1.0, size=spec.weights),
        rng.uniform(0.5, 2.0, size=spec.targets),
        rng.uniform(0.5, 2.0, size=spec.targets),
        rng.uniform(0.5, 2.0, size=spec.sources),
    )


def _positive(rng: np.random.Generator, shape: tuple[int, ...], complex_: bool) -> np.ndarray:
    values = rng.uniform(0.5, 2.0, size=shape)
    if complex_:
        values = values + 1j * rng.uniform(0.5, 2.0, size=shape)
    return values


def _signed(rng: np.random.Generator, shape: tuple[int, ...], complex_: bool) -> np.ndarray:
    values = rng.standard_normal(shape)
    if complex_:
        values = values + 1j * rng.standard_normal(shape)
    return values


_PAYLOADS = (((), False), ((2,), False), ((2,), True), ((2, 3), True))


# ---------------------------------------------------------------------------
# Forward and transpose actions against the exact chronological product.


@pytest.mark.parametrize("spec_name", list(_SPECS))
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(("payload", "complex_"), _PAYLOADS)
def test_apply_matches_exact_chronological_product(
    spec_name: str, mode: str, payload: tuple[int, ...], complex_: bool
) -> None:
    spec = _SPECS[spec_name]
    data = _data(spec, 11)
    values = _positive(np.random.default_rng(12), (spec.sources, *payload), complex_)

    got = material_chain_apply(
        _schedule(spec, spec_name),
        jnp.asarray(data.mantissas),
        jnp.asarray(data.measures),
        jnp.asarray(values),
        replay=_REPLAY,
        **{key: jnp.asarray(value) for key, value in data.endpoint(mode).items()},
    )

    expected = _exact_apply(
        spec, data.mantissas, data.measures, values, **data.endpoint(mode)
    )
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("spec_name", list(_SPECS))
@pytest.mark.parametrize("mode", ("plain", "denominators"))
@pytest.mark.parametrize(("payload", "complex_"), _PAYLOADS)
def test_transpose_matches_exact_transposed_product(
    spec_name: str, mode: str, payload: tuple[int, ...], complex_: bool
) -> None:
    spec = _SPECS[spec_name]
    data = _data(spec, 13)
    cotangents = _positive(np.random.default_rng(14), (spec.targets, *payload), complex_)

    got = material_chain_transpose(
        _schedule(spec, spec_name),
        jnp.asarray(data.mantissas),
        jnp.asarray(data.measures),
        jnp.asarray(cotangents),
        replay=_REPLAY,
        **{key: jnp.asarray(value) for key, value in data.endpoint(mode).items()},
    )

    expected = _exact_transpose(
        spec, data.mantissas, data.measures, cotangents, **data.endpoint(mode)
    )
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("spec_name", list(_SPECS))
@pytest.mark.parametrize("mode", ("plain", "denominators"))
@pytest.mark.parametrize("complex_", [False, True])
def test_apply_and_transpose_satisfy_the_dot_product_identity(
    spec_name: str, mode: str, complex_: bool
) -> None:
    spec = _SPECS[spec_name]
    schedule = _schedule(spec, spec_name)
    data = _data(spec, 15)
    rng = np.random.default_rng(16)
    values = _signed(rng, (spec.sources, 3), complex_)
    cotangents = _signed(rng, (spec.targets, 3), complex_)
    endpoint = {key: jnp.asarray(value) for key, value in data.endpoint(mode).items()}
    mantissas = jnp.asarray(data.mantissas)
    measures = jnp.asarray(data.measures)

    forward = np.asarray(
        material_chain_apply(
            schedule, mantissas, measures, jnp.asarray(values), replay=_REPLAY, **endpoint
        )
    )
    backward = np.asarray(
        material_chain_transpose(
            schedule, mantissas, measures, jnp.asarray(cotangents), replay=_REPLAY, **endpoint
        )
    )

    left = np.sum(forward * cotangents)
    right = np.sum(values * backward)
    scale = np.sum(np.abs(forward) * np.abs(cotangents))
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12 * scale)


# ---------------------------------------------------------------------------
# Exponent range.


_HUGE = 2.0**1000
_SMALL = 2.0**-1000

_RANGE_CASES = {
    # Densities reach 2**2000 after the first chunk.
    "overflow": (np.asarray([_HUGE, _HUGE]), {}),
    # Densities reach 2**-2000 after the first chunk.
    "underflow": (np.asarray([_SMALL, _SMALL]), {}),
    "mixed": (np.asarray([_SMALL, _HUGE]), {}),
    "source_measures": (
        np.asarray([1.0, 0.75]),
        {"source_measures": np.asarray([_SMALL, _SMALL])},
    ),
    "denominators": (
        np.asarray([_HUGE, _HUGE]),
        {"denominators": np.asarray([3.0 * _SMALL, _SMALL])},
    ),
}


@pytest.mark.parametrize("case", list(_RANGE_CASES))
@pytest.mark.parametrize("complex_", [False, True])
def test_compress_then_expand_is_finite_where_naive_densities_are_not(
    case: str, complex_: bool
) -> None:
    values, endpoint = _RANGE_CASES[case]
    if complex_:
        values = values + 1j * values[::-1]
    mantissas = np.random.default_rng(17).uniform(0.5, 1.0, size=_RANGE.weights)
    measures = (
        np.asarray([_SMALL, _SMALL]) if case in ("source_measures", "denominators")
        else np.asarray([0.75, 1.25])
    )

    got = np.asarray(
        material_chain_apply(
            _schedule(_RANGE, "range"),
            jnp.asarray(mantissas),
            jnp.asarray(measures),
            jnp.asarray(values),
            replay=_REPLAY,
            **{key: jnp.asarray(value) for key, value in endpoint.items()},
        )
    )

    expected = _exact_apply(_RANGE, mantissas, measures, values, **endpoint)
    assert np.all(np.isfinite(expected)) and np.all(expected != 0)
    naive = _naive_apply(_RANGE, mantissas, measures, values, **endpoint)
    assert not np.all(np.isfinite(naive) & np.isclose(naive, expected, rtol=1e-6, atol=0.0))
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("complex_", [False, True])
def test_subnormal_input_reaches_a_correctly_rounded_subnormal_output(complex_: bool) -> None:
    least = float(np.finfo(np.float64).smallest_subnormal)
    values = np.asarray([least, 0.0])
    if complex_:
        values = values * (1.0 + 1.0j)
    mantissas = np.random.default_rng(18).uniform(0.5, 1.0, size=_RANGE.weights)
    # Every end-to-end weight of the first source lies in [1/4, 1), so the
    # outputs sit near 2**-1035: subnormal with about 38 significant bits.
    measures = np.asarray([2.0**40, 2.0**40])

    got = np.asarray(
        material_chain_apply(
            _schedule(_RANGE, "range"),
            jnp.asarray(mantissas),
            jnp.asarray(measures),
            jnp.asarray(values),
            replay=_REPLAY,
        )
    )

    expected = _exact_apply(_RANGE, mantissas, measures, values)
    parts = [(got, expected)] if not complex_ else [
        (got.real, expected.real),
        (got.imag, expected.imag),
    ]
    tiny = float(np.finfo(np.float64).smallest_normal)
    for actual, exact in parts:
        assert np.all((exact > 0.0) & (exact < tiny))
        assert np.all(actual != 0.0)
        np.testing.assert_allclose(actual, exact, rtol=0.0, atol=least)


# ---------------------------------------------------------------------------
# Differentiation against the dense oracle.


_AD_SPECS = ("three", "long")


def _ad_primals(spec: _Spec, mode: str, seed: int, /) -> tuple[jax.Array, ...]:
    data = _data(spec, seed)
    values = np.random.default_rng(seed + 1).uniform(0.5, 2.0, size=(spec.sources, 2))
    return tuple(
        jnp.asarray(value)
        for value in (values, data.mantissas, data.measures, *data.extra(mode))
    )


def _tangents(primals: Sequence[jax.Array], seed: int, /) -> tuple[jax.Array, ...]:
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.uniform(0.1, 1.0, size=p.shape)) for p in primals)


def _restricted(
    function: Any, primals: Sequence[jax.Array], active: Sequence[int], /
) -> Callable[..., jax.Array]:
    def restricted(*arguments: jax.Array) -> jax.Array:
        full = list(primals)
        for index, argument in zip(active, arguments, strict=True):
            full[index] = argument
        return function(*full)

    return restricted


_ACTIVE = {
    "values": (0,),
    "mantissas": (1,),
    "measures": (2,),
    "endpoint": (3,),
    "all": None,
}


@pytest.mark.parametrize("spec_name", _AD_SPECS)
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("active_name", list(_ACTIVE))
def test_jvp_matches_dense_oracle(spec_name: str, mode: str, active_name: str) -> None:
    spec = _SPECS[spec_name]
    primals = _ad_primals(spec, mode, 21)
    active = _ACTIVE[active_name] or tuple(range(len(primals)))
    if max(active) >= len(primals):
        pytest.skip("The plain endpoint has no denominator or source measure.")
    tangents = _tangents(primals, 22)
    chosen = tuple(primals[i] for i in active)
    directions = tuple(tangents[i] for i in active)
    kernel = _restricted(_kernel(_schedule(spec, spec_name), mode), primals, active)
    oracle = _restricted(_oracle(spec, mode), primals, active)

    value, tangent = jax.jvp(kernel, chosen, directions)
    expected_value, expected_tangent = jax.jvp(oracle, chosen, directions)

    np.testing.assert_allclose(np.asarray(value), np.asarray(expected_value), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(expected_tangent), rtol=1e-12)


@pytest.mark.parametrize("spec_name", _AD_SPECS)
@pytest.mark.parametrize("mode", _MODES)
def test_vjp_matches_dense_oracle(spec_name: str, mode: str) -> None:
    spec = _SPECS[spec_name]
    primals = _ad_primals(spec, mode, 23)
    cotangent = jnp.asarray(np.random.default_rng(24).uniform(0.1, 1.0, size=(spec.targets, 2)))

    value, pullback = jax.vjp(_kernel(_schedule(spec, spec_name), mode), *primals)
    expected_value, expected_pullback = jax.vjp(_oracle(spec, mode), *primals)

    np.testing.assert_allclose(np.asarray(value), np.asarray(expected_value), rtol=1e-13)
    for got, expected in zip(pullback(cotangent), expected_pullback(cotangent), strict=True):
        assert got.shape == expected.shape
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-12)


@pytest.mark.parametrize("mode", ("plain", "denominators"))
@pytest.mark.parametrize("loss_name", ["pairing", "quadratic"])
def test_mixed_second_derivative_across_chunks(mode: str, loss_name: str) -> None:
    spec = _THREE
    primals = _ad_primals(spec, mode, 25)
    weights = jnp.asarray(np.random.default_rng(26).uniform(0.1, 1.0, size=(spec.targets, 2)))

    def loss(function: Any) -> Callable[[jax.Array], jax.Array]:
        def scalar(mantissas: jax.Array) -> jax.Array:
            y = function(primals[0], mantissas, *primals[2:])
            if loss_name == "pairing":
                return jnp.sum(weights * y)
            return 0.5 * jnp.sum(y * y) + jnp.sum(weights * y)

        return scalar

    kernel = loss(_kernel(_schedule(spec, "three"), mode))
    hessian = np.asarray(jax.jacfwd(jax.grad(kernel))(primals[1]))
    expected = np.asarray(jax.jacfwd(jax.grad(loss(_oracle(spec, mode))))(primals[1]))

    # Row 1 is chunk 0's split weight feeding chunk 1's rows 2 and 5.
    assert expected[1, 5] != 0.0 and expected[1, 2] != 0.0
    if loss_name == "pairing":
        # The pairing is linear in each chunk's weights.
        assert expected[2, 5] == 0.0
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(hessian, expected, rtol=1e-12, atol=1e-14 * scale)
    np.testing.assert_allclose(hessian[1, 5], expected[1, 5], rtol=1e-12)


def test_jvp_at_zero_values_with_nonzero_tangents() -> None:
    spec = _THREE
    mode = "denominators"
    primals = _ad_primals(spec, mode, 27)
    primals = (jnp.zeros_like(primals[0]), *primals[1:])
    tangents = _tangents(primals, 28)

    value, tangent = jax.jvp(_kernel(_schedule(spec, "three"), mode), primals, tangents)
    expected_value, expected_tangent = jax.jvp(_oracle(spec, mode), primals, tangents)

    assert np.all(np.asarray(value) == 0.0)
    np.testing.assert_array_equal(np.asarray(value), np.asarray(expected_value))
    assert np.all(np.asarray(expected_tangent) != 0.0)
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(expected_tangent), rtol=1e-12)


def test_numeric_zero_mantissa_tangent_matches_oracle() -> None:
    spec = _THREE
    mode = "plain"
    primals = _ad_primals(spec, mode, 29)
    values_tangent = _tangents(primals[:1], 30)[0]
    kernel = _kernel(_schedule(spec, "three"), mode)

    def pair(function: Any) -> Callable[[jax.Array, jax.Array], jax.Array]:
        return lambda values, mantissas: function(values, mantissas, *primals[2:])

    directions = (values_tangent, jnp.zeros_like(primals[1]))
    _, tangent = jax.jvp(pair(kernel), primals[:2], directions)
    _, expected = jax.jvp(pair(_oracle(spec, mode)), primals[:2], directions)
    _, values_only = jax.jvp(
        lambda values: kernel(values, *primals[1:]), primals[:1], (values_tangent,)
    )

    np.testing.assert_allclose(np.asarray(tangent), np.asarray(expected), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(values_only), rtol=1e-14)


@pytest.mark.parametrize("complex_", [False, True])
def test_vmap_over_value_batches(complex_: bool) -> None:
    spec = _LONG
    schedule = _schedule(spec, "long")
    data = _data(spec, 31)
    batch = _positive(np.random.default_rng(32), (3, spec.sources, 2), complex_)
    cotangents = _positive(np.random.default_rng(33), (3, spec.targets, 2), complex_)
    mantissas = jnp.asarray(data.mantissas)
    measures = jnp.asarray(data.measures)
    denominators = jnp.asarray(data.denominators)

    forward = jax.vmap(
        lambda values: material_chain_apply(
            schedule, mantissas, measures, values, denominators=denominators, replay=_REPLAY
        )
    )(jnp.asarray(batch))
    backward = jax.vmap(
        lambda values: material_chain_transpose(
            schedule, mantissas, measures, values, denominators=denominators, replay=_REPLAY
        )
    )(jnp.asarray(cotangents))

    for index in range(batch.shape[0]):
        np.testing.assert_allclose(
            np.asarray(forward[index]),
            _exact_apply(
                spec, data.mantissas, data.measures, batch[index], denominators=data.denominators
            ),
            rtol=1e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(backward[index]),
            _exact_transpose(
                spec,
                data.mantissas,
                data.measures,
                cotangents[index],
                denominators=data.denominators,
            ),
            rtol=1e-14,
            atol=0.0,
        )


# ---------------------------------------------------------------------------
# Checkpointed replay of the mantissa gradient.


def test_replay_admits_exactly_its_full_peak_and_preserves_the_mantissa_gradient() -> None:
    spec = _LONG
    schedule = _schedule(spec, "long")
    mode = "plain"
    primals = _ad_primals(spec, mode, 34)
    cotangent = jnp.asarray(np.random.default_rng(35).uniform(0.1, 1.0, size=(spec.targets, 2)))
    channels = primals[0].shape[1]

    def gradient(function: Any) -> tuple[np.ndarray, np.ndarray]:
        value, pullback = jax.vjp(
            lambda mantissas: function(primals[0], mantissas, *primals[2:]), primals[1]
        )
        (result,) = pullback(cotangent)
        return np.asarray(value), np.asarray(result)

    # The refusal reports the whole replay peak, not only its checkpoints.
    with pytest.raises(ValueError, match="checkpoint byte budget") as caught:
        gradient(_kernel(schedule, mode, (1, 1 << 30)))
    match = re.search(r"peak (\d+) bytes", str(caught.value))
    assert match is not None
    peak = int(match.group(1))
    # More than one replayed state: every slot's float64 mantissa and int32
    # exponent, plus the tape, adjoint, weight gradients and lanes.
    assert peak > 2 * spec.slots * channels * (8 + 4)
    with pytest.raises(ValueError, match="checkpoint byte budget"):
        gradient(_kernel(schedule, mode, (peak - 1, 1 << 30)))

    replayed_value, replayed = gradient(_kernel(schedule, mode, (peak, 1 << 30)))
    default_value, default = gradient(_kernel(schedule, mode, _REPLAY))
    _, expected = gradient(_oracle(spec, mode))
    np.testing.assert_array_equal(replayed_value, default_value)
    np.testing.assert_array_equal(replayed, default)
    np.testing.assert_allclose(replayed, expected, rtol=1e-12)


def test_compiled_replay_temporaries_stay_within_the_admitted_peak() -> None:
    # Independent of the peak model: the compiler's own buffer assignment for
    # the replayed mantissa pullback, admitted exactly at its reported peak.
    spec = _LONG
    schedule = _schedule(spec, "long")
    mode = "plain"
    primals = _ad_primals(spec, mode, 34)
    cotangent = jnp.ones((spec.targets, primals[0].shape[1]))

    def pullback(replay: tuple[int, int]) -> Callable[..., jax.Array]:
        kernel = _kernel(schedule, mode, replay)

        def run(mantissas: jax.Array, ct: jax.Array) -> jax.Array:
            _, vjp = jax.vjp(lambda m: kernel(primals[0], m, *primals[2:]), mantissas)
            return vjp(ct)[0]

        return run

    with pytest.raises(ValueError, match="checkpoint byte budget") as caught:
        jax.jit(pullback((1, 1 << 30))).lower(primals[1], cotangent)
    match = re.search(r"peak (\d+) bytes", str(caught.value))
    assert match is not None
    peak = int(match.group(1))
    compiled = jax.jit(pullback((peak, 1 << 30))).lower(primals[1], cotangent).compile()
    analysis = compiled.memory_analysis()
    if analysis is None:
        pytest.skip("this backend reports no compiled memory analysis")
    assert analysis.temp_size_in_bytes <= peak


def test_replay_refuses_more_chunk_updates_than_its_operation_budget() -> None:
    spec = _LONG
    schedule = _schedule(spec, "long")
    primals = _ad_primals(spec, "plain", 34)
    cotangent = jnp.ones((spec.targets, primals[0].shape[1]))
    # A forward pass plus one reverse recomputation executes 2 * chunks updates.
    starved = (1 << 30, len(spec.chunks))
    _, pullback = jax.vjp(
        lambda mantissas: _kernel(schedule, "plain", starved)(
            primals[0], mantissas, *primals[2:]
        ),
        primals[1],
    )
    with pytest.raises(ValueError, match="operation budget"):
        pullback(cotangent)

"""Element-by-element executable specification of native interval instructions.

This test oracle restates ``phydrax.geometry._interval_native`` semantics with
scalar IEEE operations exactly as the meshcore kernel must perform them. It is
compared bitwise with the canonical NumPy interval interpreter, never used by
production code.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from phydrax.geometry import _interval_native as native
from phydrax.geometry._interval_enclosure import NativeIntervalProgram


_EPSILON = float(np.finfo(np.float64).eps)
_TWO_PI = 2.0 * math.pi
_TRANSCENDENTAL_STEPS = 4


def _minimum(first: float, second: float, /) -> float:
    """NumPy ``minimum``: the first NaN operand, ``-0`` below ``+0``."""
    if math.isnan(first):
        return first
    if math.isnan(second):
        return second
    if first == second:
        return first if math.copysign(1.0, first) < 0 else second
    return first if first < second else second


def _maximum(first: float, second: float, /) -> float:
    if math.isnan(first):
        return first
    if math.isnan(second):
        return second
    if first == second:
        return second if math.copysign(1.0, first) < 0 else first
    return first if first > second else second


def _down(value: float, steps: int = 1, /) -> float:
    for _ in range(steps):
        value = math.nextafter(value, -math.inf)
    return value


def _up(value: float, steps: int = 1, /) -> float:
    for _ in range(steps):
        value = math.nextafter(value, math.inf)
    return value


def _product(first: float, second: float, /) -> float:
    if (first == 0.0 and math.isinf(second)) or (second == 0.0 and math.isinf(first)):
        return 0.0
    return first * second


def _contains_phase(lower: float, upper: float, phase: float, /) -> bool:
    slack = 8.0 * _EPSILON * _maximum(1.0, abs(lower))
    index = float(np.ceil((lower - slack - phase) / _TWO_PI))
    return phase + index * _TWO_PI <= upper + slack


def _trigonometric(
    lower: float,
    upper: float,
    function: Callable[[float], float],
    high_phase: float,
    low_phase: float,
    /,
) -> tuple[float, float]:
    wide = not (upper - lower < _TWO_PI) or not math.isfinite(lower) or not math.isfinite(upper)
    first, last = function(lower), function(upper)
    low, high = _minimum(first, last), _maximum(first, last)
    if _contains_phase(lower, upper, high_phase) or wide:
        high = 1.0
    if _contains_phase(lower, upper, low_phase) or wide:
        low = -1.0
    low, high = _down(low, _TRANSCENDENTAL_STEPS), _up(high, _TRANSCENDENTAL_STEPS)
    return _maximum(low, -1.0), _minimum(high, 1.0)


def _sine(value: float, /) -> float:
    return math.sin(value) if math.isfinite(value) else math.nan


def _cosine(value: float, /) -> float:
    return math.cos(value) if math.isfinite(value) else math.nan


def execute(
    program: NativeIntervalProgram, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Execute one batch: ``lower``/``upper`` have the program input shape."""
    low = [float(value) for value in program.template_lower]
    high = [float(value) for value in program.template_upper]
    for position, (first, last) in enumerate(
        zip(np.asarray(lower).reshape(-1), np.asarray(upper).reshape(-1), strict=True)
    ):
        low[program.input_offset + position] = float(first)
        high[program.input_offset + position] = float(last)
    indices = [int(value) for value in program.indices]

    def address(offset: int, index: int, element: int, /) -> int:
        return offset + (element if index < 0 else indices[index + element])

    for row in program.instructions:
        opcode, out, size, first, first_index, second, second_index, auxiliary = (
            int(value) for value in row
        )
        for element in range(size):
            target = out + element
            if opcode == native.REDUCE_SUM:
                base = first + element * auxiliary
                lower_sum = upper_sum = scale = 0.0
                for term in range(auxiliary):
                    lower_sum += low[base + term]
                    upper_sum += high[base + term]
                for term in range(auxiliary):
                    scale += _maximum(abs(low[base + term]), abs(high[base + term]))
                gamma = float(program.parameters[second_index])
                low[target] = _down(lower_sum - gamma * scale)
                high[target] = _up(upper_sum + gamma * scale)
                continue
            a = address(first, first_index, element)
            a_low, a_high = low[a], high[a]
            if opcode == native.GATHER:
                low[target], high[target] = a_low, a_high
            elif opcode == native.NEG:
                low[target], high[target] = -a_high, -a_low
            elif opcode in (native.SIN, native.COS):
                low[target], high[target] = _trigonometric(
                    a_low,
                    a_high,
                    _sine if opcode == native.SIN else _cosine,
                    0.5 * math.pi if opcode == native.SIN else 0.0,
                    -0.5 * math.pi if opcode == native.SIN else math.pi,
                )
            else:
                b = address(second, second_index, element)
                b_low, b_high = low[b], high[b]
                if opcode == native.ADD:
                    low[target], high[target] = _down(a_low + b_low), _up(a_high + b_high)
                elif opcode in (native.SUB, native.SUB_SAME):
                    result_low, result_high = _down(a_low - b_high), _up(a_high - b_low)
                    if a_low == a_high and b_low == b_high and a_low == b_low and math.isfinite(a_low):
                        result_low = result_high = 0.0
                    if opcode == native.SUB_SAME and math.isfinite(a_low) and math.isfinite(a_high):
                        result_low = result_high = 0.0
                    low[target], high[target] = result_low, result_high
                elif opcode == native.MUL:
                    ll, lu = _product(a_low, b_low), _product(a_low, b_high)
                    ul, uu = _product(a_high, b_low), _product(a_high, b_high)
                    low[target] = _down(_minimum(_minimum(ll, lu), _minimum(ul, uu)))
                    high[target] = _up(_maximum(_maximum(ll, lu), _maximum(ul, uu)))
                else:
                    raise ValueError(f"native interval opcode is unsupported; got {opcode}")
    size = int(np.prod(program.output_shape))
    window = slice(program.output_offset, program.output_offset + size)
    return (
        np.asarray(low[window], dtype=np.float64).reshape(program.output_shape),
        np.asarray(high[window], dtype=np.float64).reshape(program.output_shape),
    )

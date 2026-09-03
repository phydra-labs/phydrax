#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import keyword
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class _Axis:
    name: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Singleton:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Ellipsis:
    start: int
    end: int


_Factor = _Axis | _Singleton | _Ellipsis


@dataclass(frozen=True, slots=True)
class _PhysicalAxis:
    factors: tuple[_Factor, ...]
    grouped: bool
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Expression:
    axes: tuple[_PhysicalAxis, ...]


@dataclass(frozen=True, slots=True)
class _Pattern:
    source: str
    left: _Expression
    right: _Expression


def _pattern_error(source: str, position: int, message: str) -> ValueError:
    position = min(max(position, 0), len(source))
    return ValueError(f"Invalid ein pattern: {message}\n  {source}\n  {' ' * position}^")


def _is_ascii_letter(char: str) -> bool:
    return "a" <= char <= "z" or "A" <= char <= "Z"


def _is_name_char(char: str) -> bool:
    return _is_ascii_letter(char) or char.isdigit() or char == "_"


def _skip_whitespace(source: str, position: int, end: int) -> int:
    while position < end and source[position].isspace():
        position += 1
    return position


def _parse_factor(source: str, position: int, end: int) -> tuple[_Factor, int]:
    if source.startswith("...", position):
        return _Ellipsis(position, position + 3), position + 3

    char = source[position]
    if _is_ascii_letter(char):
        token_end = position + 1
        while token_end < end and _is_name_char(source[token_end]):
            token_end += 1
        name = source[position:token_end]
        if keyword.iskeyword(name):
            raise _pattern_error(
                source,
                position,
                f"axis name {name!r} is a Python keyword",
            )
        return _Axis(name, position, token_end), token_end

    if char.isdigit():
        token_end = position + 1
        while token_end < end and _is_name_char(source[token_end]):
            token_end += 1
        token = source[position:token_end]
        if token == "1":
            return _Singleton(position, token_end), token_end
        raise _pattern_error(
            source,
            position,
            f"only anonymous literal '1' is supported, got {token!r}",
        )

    if char == "_":
        raise _pattern_error(
            source,
            position,
            "axis names must start with an ASCII letter",
        )

    raise _pattern_error(source, position, f"unexpected character {char!r}")


def _parse_group(source: str, position: int, end: int) -> tuple[_PhysicalAxis, int]:
    group_start = position
    position = _skip_whitespace(source, position + 1, end)
    factors: list[_Factor] = []

    while position < end and source[position] != ")":
        if source[position] == "(":
            raise _pattern_error(source, position, "nested groups are not supported")
        factor, position = _parse_factor(source, position, end)
        factors.append(factor)
        if position < end and source[position] != ")" and not source[position].isspace():
            raise _pattern_error(
                source,
                position,
                "group factors must be separated by whitespace",
            )
        position = _skip_whitespace(source, position, end)

    if position >= end:
        raise _pattern_error(source, group_start, "unclosed group")
    if not factors:
        raise _pattern_error(
            source, group_start, "empty groups are not supported; use '1'"
        )

    return (
        _PhysicalAxis(tuple(factors), True, group_start, position + 1),
        position + 1,
    )


def _parse_expression(source: str, start: int, end: int) -> _Expression:
    axes: list[_PhysicalAxis] = []
    position = _skip_whitespace(source, start, end)

    while position < end:
        axis_start = position
        if source[position] == "(":
            physical_axis, position = _parse_group(source, position, end)
        elif source[position] == ")":
            raise _pattern_error(source, position, "unexpected closing parenthesis")
        else:
            factor, position = _parse_factor(source, position, end)
            physical_axis = _PhysicalAxis(
                (factor,),
                False,
                axis_start,
                position,
            )
        axes.append(physical_axis)

        if position < end and not source[position].isspace():
            raise _pattern_error(
                source,
                position,
                "physical axes must be separated by whitespace",
            )
        position = _skip_whitespace(source, position, end)

    return _Expression(tuple(axes))


def _axis_tokens(expression: _Expression) -> tuple[_Axis, ...]:
    return tuple(
        factor
        for physical_axis in expression.axes
        for factor in physical_axis.factors
        if isinstance(factor, _Axis)
    )


def _ellipsis_tokens(expression: _Expression) -> tuple[_Ellipsis, ...]:
    return tuple(
        factor
        for physical_axis in expression.axes
        for factor in physical_axis.factors
        if isinstance(factor, _Ellipsis)
    )


@lru_cache(maxsize=256)
def _parse_pattern(source: str) -> _Pattern:
    if not isinstance(source, str):
        raise TypeError(f"ein pattern must be a string, got {type(source).__name__}")

    arrow_count = source.count("->")
    if arrow_count != 1:
        position = source.find("->")
        if position < 0:
            position = len(source)
        raise _pattern_error(
            source,
            position,
            f"expected exactly one '->', found {arrow_count}",
        )

    arrow = source.index("->")
    left = _parse_expression(source, 0, arrow)
    right = _parse_expression(source, arrow + 2, len(source))

    for expression, side in ((left, "input"), (right, "output")):
        names: set[str] = set()
        for axis in _axis_tokens(expression):
            if axis.name in names:
                raise _pattern_error(
                    source,
                    axis.start,
                    f"axis {axis.name!r} appears more than once on the {side} side",
                )
            names.add(axis.name)

        ellipses = _ellipsis_tokens(expression)
        if len(ellipses) > 1:
            raise _pattern_error(
                source,
                ellipses[1].start,
                f"at most one ellipsis is allowed on the {side} side",
            )

    for physical_axis in left.axes:
        for factor in physical_axis.factors:
            if isinstance(factor, _Ellipsis) and physical_axis.grouped:
                raise _pattern_error(
                    source,
                    factor.start,
                    "input ellipsis must be a standalone physical axis",
                )

    left_ellipsis = _ellipsis_tokens(left)
    right_ellipsis = _ellipsis_tokens(right)
    if right_ellipsis and not left_ellipsis:
        raise _pattern_error(
            source,
            right_ellipsis[0].start,
            "output ellipsis requires an input ellipsis",
        )

    return _Pattern(source, left, right)

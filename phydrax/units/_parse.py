"""Exact resolution of unit expressions over the catalog symbol table."""

from __future__ import annotations

import re
from fractions import Fraction

from phydrax.units import _catalog
from phydrax.units._catalog import ONE
from phydrax.units._unit import derived_unit, UnitComponent, UnitDefinition


_ATOMIC_SYMBOL = re.compile(r"[A-Za-z_]+")
_TOKEN = re.compile(
    r"(?P<space>\s+)"
    r"|(?P<symbol>[A-Za-z_]+)(?P<suffix>[0-9]*)"
    r"|(?P<integer>[0-9]+)"
    r"|(?P<operator>[*/^()+-])"
)

# One token: (kind, text, integer suffix, preceded by whitespace).
_Token = tuple[str, str, int | None, bool]


def _symbol_table() -> dict[str, UnitDefinition]:
    # Atomic catalog symbols are the grammar's terminals; every compound catalog
    # symbol (``m3``, ``kg/m3``, ``cm^-1``) is re-derived exactly by the grammar.
    table: dict[str, UnitDefinition] = {}
    for value in vars(_catalog).values():
        if not isinstance(value, UnitDefinition):
            continue
        if not _ATOMIC_SYMBOL.fullmatch(value.symbol):
            continue
        if table.setdefault(value.symbol, value) != value:
            raise ValueError(f"Unit catalog symbol {value.symbol!r} is ambiguous.")
    return table


_SYMBOLS = _symbol_table()


def _tokens(expression: str, /) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    spaced = False
    position = 0
    while position < len(expression):
        match = _TOKEN.match(expression, position)
        if match is None:
            raise ValueError(
                f"Unit expression {expression!r} contains an unsupported character "
                f"at position {position}."
            )
        position = match.end()
        if match.group("space"):
            spaced = True
            continue
        if match.group("symbol"):
            suffix = match.group("suffix")
            tokens.append(
                ("symbol", match.group("symbol"), int(suffix) if suffix else None, spaced)
            )
        elif match.group("integer"):
            tokens.append(("integer", match.group("integer"), None, spaced))
        else:
            tokens.append(("operator", match.group("operator"), None, spaced))
        spaced = False
    return tuple(tokens)


class _Parser:
    """Recursive-descent parser producing exact ``(unit, exponent)`` factors.

    Grammar (products and quotients associate left to right)::

        product  := power (("*" | "/" | whitespace) power)*
        power    := primary ["^" exponent]
        primary  := symbol[integer suffix] | "1" | "(" product ")"
        exponent := [+-] integer ["/" integer] | "(" exponent ")"
    """

    def __init__(self, expression: str, tokens: tuple[_Token, ...], /):
        self.expression = expression
        self.tokens = tokens
        self.position = 0

    def fail(self, reason: str, /) -> ValueError:
        return ValueError(f"Unit expression {self.expression!r} {reason}.")

    def peek(self) -> _Token | None:
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def is_operator(self, text: str, /) -> bool:
        token = self.peek()
        return token is not None and token[0] == "operator" and token[1] == text

    def expect_operator(self, text: str, /) -> None:
        if not self.is_operator(text):
            raise self.fail(f"expects {text!r}")
        self.position += 1

    def integer(self) -> int:
        token = self.peek()
        if token is None or token[0] != "integer":
            raise self.fail("expects an integer exponent")
        self.position += 1
        return int(token[1])

    def parse(self) -> tuple[UnitComponent, ...]:
        factors = self.product()
        if self.peek() is not None:
            raise self.fail(f"has an unexpected {self.peek()[1]!r}")
        return factors

    def product(self) -> tuple[UnitComponent, ...]:
        factors = list(self.power())
        divided = False
        while (token := self.peek()) is not None and not self.is_operator(")"):
            if token[0] == "operator" and token[1] in "*/":
                self.position += 1
                sign = Fraction(-1) if token[1] == "/" else Fraction(1)
                divided = divided or token[1] == "/"
            elif token[3]:
                # "J/kg K" reads as J/(kg K) to some and (J/kg) K to others.
                if divided:
                    raise self.fail(
                        "mixes a quotient with a whitespace product; use parentheses"
                    )
                sign = Fraction(1)
            else:
                raise self.fail(f"has an unexpected {token[1]!r}")
            factors.extend((unit, sign * power) for unit, power in self.power())
        return tuple(factors)

    def power(self) -> tuple[UnitComponent, ...]:
        factors = self.primary()
        if not self.is_operator("^"):
            return factors
        self.position += 1
        if self.is_operator("("):
            self.position += 1
            exponent = self.exponent()
            self.expect_operator(")")
        else:
            exponent = self.exponent()
        return tuple((unit, power * exponent) for unit, power in factors)

    def exponent(self) -> Fraction:
        negative = self.is_operator("-")
        if negative or self.is_operator("+"):
            self.position += 1
        numerator = self.integer()
        denominator = 1
        if self.is_operator("/"):
            following = self.tokens[self.position + 1 : self.position + 2]
            if following and following[0][0] == "integer":
                self.position += 1
                denominator = self.integer()
        if denominator == 0:
            raise self.fail("has a zero exponent denominator")
        value = Fraction(numerator, denominator)
        return -value if negative else value

    def primary(self) -> tuple[UnitComponent, ...]:
        token = self.peek()
        if token is None:
            raise self.fail("ends where a unit is expected")
        self.position += 1
        kind, text, suffix, _ = token
        if kind == "symbol":
            unit = _SYMBOLS.get(text)
            if unit is None:
                raise self.fail(f"uses unknown unit symbol {text!r}")
            return ((unit, Fraction(1 if suffix is None else suffix)),)
        if kind == "integer" and text == "1":
            return ((ONE, Fraction(1)),)
        if kind == "operator" and text == "(":
            factors = self.product()
            self.expect_operator(")")
            return factors
        raise self.fail(f"has an unexpected {text!r} where a unit is expected")


def parse_unit(expression: str, /) -> UnitDefinition:
    """Resolve a unit expression built from catalog symbols into an exact unit.

    Terminals are the atomic catalog symbols (``m``, ``kg``, ``Pa``, ``Hz``, ...)
    and ``1`` for the dimensionless unit. A symbol may carry an unsigned integer
    power suffix (``m3``, ``s2``). Factors combine with ``*``, ``/``, or
    whitespace, left to right; ``^`` raises a unit or parenthesized group to an
    integer or rational power (``m^-1``, ``J^-1/2``, ``cm^(1/2)``). Whitespace
    products after a quotient in the same group are rejected as ambiguous.

    The result carries ``expression`` as its symbol; its dimension and exact
    scale follow from the factors, so every catalog unit resolves from its own
    symbol to an equal `UnitDefinition`. Unknown symbols, malformed syntax, and
    rational powers without an exact rational scale raise `ValueError` naming
    the expression.
    """
    if not isinstance(expression, str):
        raise TypeError("Unit expressions must be strings.")
    if not expression or expression.strip() != expression:
        raise ValueError(
            f"Unit expression {expression!r} must be non-empty without surrounding space."
        )
    factors = _Parser(expression, _tokens(expression)).parse()
    try:
        return derived_unit(expression, factors)
    except ValueError as error:
        raise ValueError(
            f"Unit expression {expression!r} has no exact rational scale: {error}."
        ) from error


__all__ = ["parse_unit"]

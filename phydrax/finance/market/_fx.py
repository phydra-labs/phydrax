#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..core import Currency
from ._risk_factors import MarketState, RiskFactorKey
from ._status import MarketStatus


class FXConversionResult(StrictModule):
    values: Array
    valid: Array
    status: Array
    path_id: str = eqx.field(static=True)


class FXConversionPath(StrictModule, NonTrainableState):
    """One explicit directed sequence of quoted FX factors."""

    source_currency: Currency = eqx.field(static=True)
    target_currency: Currency = eqx.field(static=True)
    factor_keys: tuple[RiskFactorKey, ...]
    directions: tuple[int, ...] = eqx.field(static=True)
    route_codes: tuple[str, ...] = eqx.field(static=True)
    path_status: MarketStatus = eqx.field(static=True)
    path_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_currency: Currency,
        target_currency: Currency,
        factor_keys: Sequence[RiskFactorKey],
        directions: Sequence[int],
        /,
    ):
        if not isinstance(source_currency, Currency) or not isinstance(
            target_currency, Currency
        ):
            raise TypeError("FX path endpoints must be Currency values.")
        factors = tuple(factor_keys)
        direction_values = tuple(directions)
        if len(factors) != len(direction_values):
            raise ValueError("factor_keys and directions must have equal length.")
        if not all(isinstance(value, RiskFactorKey) for value in factors):
            raise TypeError("factor_keys must contain RiskFactorKey values.")
        if any(value not in (-1, 1) for value in direction_values):
            raise ValueError("FX path directions must be +1 or -1.")
        if any(value.quote_key.fx_pair is None for value in factors):
            raise ValueError("Every FX conversion factor must reference an FXPair quote.")
        current = source_currency.currency_id
        route = [source_currency.code]
        status = MarketStatus.SUCCESS
        for factor, direction in zip(factors, direction_values, strict=True):
            pair = factor.quote_key.fx_pair
            if pair is None:
                raise AssertionError("validated FX pair unexpectedly missing")
            expected = pair.base if direction == 1 else pair.quote
            following = pair.quote if direction == 1 else pair.base
            if expected.currency_id != current:
                status |= MarketStatus.CURRENCY_MISMATCH
            current = following.currency_id
            route.append(following.code)
        if current != target_currency.currency_id:
            status |= MarketStatus.CURRENCY_MISMATCH
        if not factors and source_currency.currency_id != target_currency.currency_id:
            status |= MarketStatus.CURRENCY_MISMATCH
        self.source_currency = source_currency
        self.target_currency = target_currency
        self.factor_keys = factors
        self.directions = direction_values
        self.route_codes = tuple(route)
        self.path_status = status
        self.path_id = canonical_fingerprint(
            {
                "kind": "financial-fx-conversion-path",
                "source_currency_id": source_currency.currency_id,
                "target_currency_id": target_currency.currency_id,
                "factor_ids": [value.factor_id for value in factors],
                "directions": list(direction_values),
                "route_codes": route,
                "status": int(status),
            }
        )

    def convert(
        self,
        amounts: ArrayLike,
        market: MarketState,
        /,
        *,
        source_currency: Currency,
        target_currency: Currency,
    ) -> FXConversionResult:
        """Apply this route only when its endpoints and every factor are accepted."""

        if not isinstance(market, MarketState):
            raise TypeError("market must be a MarketState.")
        if not isinstance(source_currency, Currency) or not isinstance(
            target_currency, Currency
        ):
            raise TypeError("conversion endpoints must be Currency values.")
        values = jnp.asarray(amounts)
        if not (
            jnp.issubdtype(values.dtype, jnp.integer)
            or jnp.issubdtype(values.dtype, jnp.floating)
        ):
            raise TypeError("amounts must contain real numeric values.")
        finite = jnp.all(jnp.isfinite(values))
        status = jnp.asarray(int(self.path_status), dtype=jnp.int32)
        endpoints_match = (
            source_currency.currency_id == self.source_currency.currency_id
            and target_currency.currency_id == self.target_currency.currency_id
        )
        if not endpoints_match:
            status = status | int(MarketStatus.CURRENCY_MISMATCH)
        result = values
        factor_valid = jnp.asarray(True)
        for factor, direction in zip(self.factor_keys, self.directions, strict=True):
            if factor.factor_id not in market.layout.factor_ids:
                status = status | int(MarketStatus.MISSING_FACTOR)
                factor_valid = jnp.asarray(False)
                continue
            index = market.layout.index(factor)
            rate = market.values[index]
            accepted = market.accepted[index] & jnp.isfinite(rate) & (rate > 0.0)
            factor_valid = factor_valid & accepted
            status = status | jnp.where(
                market.accepted[index],
                jnp.asarray(0, dtype=jnp.int32),
                market.status[index],
            )
            status = status | jnp.where(
                rate > 0.0,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(int(MarketStatus.INVALID_DOMAIN), dtype=jnp.int32),
            )
            safe_rate = jnp.where(accepted, rate, 1.0)
            result = result * safe_rate if direction == 1 else result / safe_rate
        status = status | jnp.where(
            finite,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(MarketStatus.NONFINITE), dtype=jnp.int32),
        )
        valid = finite & factor_valid & (status == int(MarketStatus.SUCCESS))
        result = jnp.where(valid, result, jnp.zeros_like(result))
        return FXConversionResult(result, valid, status, self.path_id)


class FXPathResolution(StrictModule, NonTrainableState):
    path: FXConversionPath | None
    valid: Array
    status: Array
    shortest_path_count: int = eqx.field(static=True)


class FXConversionGraph(StrictModule, NonTrainableState):
    """Host-side FX route graph that rejects ambiguous shortest paths."""

    factor_keys: tuple[RiskFactorKey, ...]
    graph_id: str = eqx.field(static=True)

    def __init__(self, factor_keys: Sequence[RiskFactorKey], /):
        factors = tuple(factor_keys)
        if not factors or not all(isinstance(value, RiskFactorKey) for value in factors):
            raise TypeError("factor_keys must contain at least one RiskFactorKey.")
        if any(value.quote_key.fx_pair is None for value in factors):
            raise ValueError("An FX conversion graph accepts only FXPair quote factors.")
        if len({value.factor_id for value in factors}) != len(factors):
            raise ValueError("FX conversion graph factors must be unique.")
        ordered = tuple(sorted(factors, key=lambda value: value.factor_id))
        self.factor_keys = ordered
        self.graph_id = canonical_fingerprint(
            {
                "kind": "financial-fx-conversion-graph",
                "factor_ids": [value.factor_id for value in ordered],
            }
        )

    def resolve(
        self,
        source_currency: Currency,
        target_currency: Currency,
        /,
        *,
        max_hops: int = 4,
    ) -> FXPathResolution:
        if not isinstance(source_currency, Currency) or not isinstance(
            target_currency, Currency
        ):
            raise TypeError("FX route endpoints must be Currency values.")
        if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1:
            raise ValueError("max_hops must be a positive integer.")
        if source_currency.currency_id == target_currency.currency_id:
            path = FXConversionPath(source_currency, target_currency, (), ())
            return FXPathResolution(
                path,
                jnp.asarray(True),
                jnp.asarray(int(MarketStatus.SUCCESS), dtype=jnp.int32),
                1,
            )
        frontier: list[
            tuple[str, tuple[RiskFactorKey, ...], tuple[int, ...], tuple[str, ...]]
        ] = [(source_currency.currency_id, (), (), (source_currency.currency_id,))]
        solutions: list[tuple[tuple[RiskFactorKey, ...], tuple[int, ...]]] = []
        for _ in range(max_hops):
            next_frontier: list[
                tuple[str, tuple[RiskFactorKey, ...], tuple[int, ...], tuple[str, ...]]
            ] = []
            for current, factors, directions, visited in frontier:
                for factor in self.factor_keys:
                    pair = factor.quote_key.fx_pair
                    if pair is None:
                        raise AssertionError("validated FX pair unexpectedly missing")
                    if pair.base.currency_id == current:
                        following, direction = pair.quote.currency_id, 1
                    elif pair.quote.currency_id == current:
                        following, direction = pair.base.currency_id, -1
                    else:
                        continue
                    if following in visited:
                        continue
                    candidate_factors = factors + (factor,)
                    candidate_directions = directions + (direction,)
                    if following == target_currency.currency_id:
                        solutions.append((candidate_factors, candidate_directions))
                    else:
                        next_frontier.append(
                            (
                                following,
                                candidate_factors,
                                candidate_directions,
                                visited + (following,),
                            )
                        )
            if solutions:
                break
            frontier = next_frontier
            if not frontier:
                break
        if not solutions:
            return FXPathResolution(
                None,
                jnp.asarray(False),
                jnp.asarray(int(MarketStatus.MISSING_FACTOR), dtype=jnp.int32),
                0,
            )
        ordered = sorted(
            solutions,
            key=lambda value: tuple(item.factor_id for item in value[0]),
        )
        if len(ordered) > 1:
            return FXPathResolution(
                None,
                jnp.asarray(False),
                jnp.asarray(int(MarketStatus.FX_AMBIGUOUS), dtype=jnp.int32),
                len(ordered),
            )
        factor_path, directions = ordered[0]
        path = FXConversionPath(
            source_currency,
            target_currency,
            factor_path,
            directions,
        )
        return FXPathResolution(
            path,
            jnp.asarray(path.path_status == MarketStatus.SUCCESS),
            jnp.asarray(int(path.path_status), dtype=jnp.int32),
            1,
        )


class FXTriangleResult(StrictModule):
    implied_cross: Array
    quoted_cross: Array
    relative_error: Array
    valid: Array
    consistent: Array
    status: Array


def fx_triangle_consistency(
    market: MarketState,
    first_leg: RiskFactorKey,
    second_leg: RiskFactorKey,
    cross: RiskFactorKey,
    /,
    *,
    relative_tolerance: float = 1.0e-8,
) -> FXTriangleResult:
    """Audit ``A/B × B/C = A/C`` without silently choosing a conversion route."""

    if not isinstance(market, MarketState):
        raise TypeError("market must be a MarketState.")
    factors = (first_leg, second_leg, cross)
    if not all(isinstance(value, RiskFactorKey) for value in factors):
        raise TypeError("FX triangle legs must be RiskFactorKey values.")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and nonnegative.")
    pairs = tuple(value.quote_key.fx_pair for value in factors)
    if any(value is None for value in pairs):
        raise ValueError("FX triangle legs must reference FXPair quotes.")
    first_pair, second_pair, cross_pair = pairs
    if first_pair is None or second_pair is None or cross_pair is None:
        raise AssertionError("validated FX pair unexpectedly missing")
    structural = (
        first_pair.quote.currency_id == second_pair.base.currency_id
        and first_pair.base.currency_id == cross_pair.base.currency_id
        and second_pair.quote.currency_id == cross_pair.quote.currency_id
    )
    indices = tuple(
        market.layout.index(value) if value.factor_id in market.layout.factor_ids else -1
        for value in factors
    )
    present = all(value >= 0 for value in indices)
    if present:
        a, b, c = (market.values[value] for value in indices)
        accepted = jnp.all(jnp.stack(tuple(market.accepted[value] for value in indices)))
    else:
        a = b = c = jnp.asarray(0.0)
        accepted = jnp.asarray(False)
    implied = a * b
    denominator = jnp.maximum(jnp.abs(c), jnp.asarray(1.0e-30, dtype=c.dtype))
    error = jnp.abs(implied - c) / denominator
    tolerance = jnp.asarray(relative_tolerance, dtype=error.dtype)
    consistent = accepted & structural & (error <= tolerance)
    status = jnp.asarray(int(MarketStatus.SUCCESS), dtype=jnp.int32)
    if not present:
        status = status | int(MarketStatus.MISSING_FACTOR)
    if not structural:
        status = status | int(MarketStatus.CURRENCY_MISMATCH)
    status = status | jnp.where(
        accepted | ~jnp.asarray(present),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(int(MarketStatus.INVALID_DOMAIN), dtype=jnp.int32),
    )
    status = status | jnp.where(
        consistent | ~accepted,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(int(MarketStatus.INVALID_DOMAIN), dtype=jnp.int32),
    )
    return FXTriangleResult(implied, c, error, accepted & structural, consistent, status)


__all__ = [
    "FXConversionGraph",
    "FXConversionPath",
    "FXConversionResult",
    "FXPathResolution",
    "FXTriangleResult",
    "fx_triangle_consistency",
]

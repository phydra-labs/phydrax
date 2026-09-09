#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed single- and multi-curve bootstrap plans with auditable replay."""

from __future__ import annotations

from math import isfinite
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import AbstractAttribute, StrictModule
from ...linalg import DenseLinearOperator, FactorizationPolicy, factorize, RankPolicy
from ...optim import (
    least_squares,
    LevenbergMarquardt,
    OptimizationTermination,
)
from ._core import (
    CurveDefinition,
    CurveRepresentation,
    CurveSet,
    PreparedCurve,
)


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be a non-empty string.")
    return identifier


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _time(value: float, name: str, /) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return number


def _schedule_arrays(
    start_times: ArrayLike,
    end_times: ArrayLike,
    payment_times: ArrayLike,
    accrual_fractions: ArrayLike,
    valid: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    start = jnp.asarray(start_times)
    end = jnp.asarray(end_times)
    payment = jnp.asarray(payment_times)
    accrual = jnp.asarray(accrual_fractions)
    mask = jnp.asarray(valid, dtype=bool)
    if not all(array.ndim == 1 for array in (start, end, payment, accrual, mask)):
        raise ValueError("Bootstrap schedule arrays must be rank one.")
    if not (start.shape == end.shape == payment.shape == accrual.shape == mask.shape):
        raise ValueError("Bootstrap schedule arrays must have identical shapes.")
    concrete = tuple(np.asarray(array) for array in (start, end, payment, accrual, mask))
    start_np, end_np, payment_np, accrual_np, mask_np = concrete
    active = mask_np.astype(bool)
    if np.any(~np.isfinite(start_np[active])) or np.any(start_np[active] < 0.0):
        raise ValueError("Active schedule start times must be finite and nonnegative.")
    if np.any(~np.isfinite(end_np[active])) or np.any(end_np[active] <= start_np[active]):
        raise ValueError(
            "Active schedule end times must be finite and later than starts."
        )
    if np.any(~np.isfinite(payment_np[active])) or np.any(
        payment_np[active] < end_np[active]
    ):
        raise ValueError("Active payment times must be finite and no earlier than ends.")
    if np.any(~np.isfinite(accrual_np[active])) or np.any(accrual_np[active] <= 0.0):
        raise ValueError("Active accrual fractions must be finite and positive.")
    return start, end, payment, accrual, mask


class DepositBootstrapInstrument(StrictModule):
    """Simple-compounded deposit quote equation."""

    instrument_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    end_time: float = eqx.field(static=True)
    accrual_fraction: float = eqx.field(static=True)
    quote_weight: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        instrument_id: str,
        discount_curve_id: str,
        start_time: float,
        end_time: float,
        accrual_fraction: float,
        quote_weight: float = 1.0,
    ):
        start = _time(start_time, "start_time")
        end = _time(end_time, "end_time")
        if end <= start:
            raise ValueError("end_time must be later than start_time.")
        self.instrument_id = _identifier(instrument_id, "instrument_id")
        self.discount_curve_id = _identifier(discount_curve_id, "discount_curve_id")
        self.start_time = start
        self.end_time = end
        self.accrual_fraction = _positive(accrual_fraction, "accrual_fraction")
        self.quote_weight = _positive(quote_weight, "quote_weight")


class ZeroRateBootstrapInstrument(StrictModule):
    """Continuously compounded zero-rate quote equation."""

    instrument_id: str = eqx.field(static=True)
    curve_id: str = eqx.field(static=True)
    maturity: float = eqx.field(static=True)
    quote_weight: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        instrument_id: str,
        curve_id: str,
        maturity: float,
        quote_weight: float = 1.0,
    ):
        maturity_ = _positive(maturity, "maturity")
        self.instrument_id = _identifier(instrument_id, "instrument_id")
        self.curve_id = _identifier(curve_id, "curve_id")
        self.maturity = maturity_
        self.quote_weight = _positive(quote_weight, "quote_weight")


class ForwardRateBootstrapInstrument(StrictModule):
    """Simple forward-rate quote equation for a projection curve."""

    instrument_id: str = eqx.field(static=True)
    projection_curve_id: str = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    end_time: float = eqx.field(static=True)
    accrual_fraction: float = eqx.field(static=True)
    quote_weight: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        instrument_id: str,
        projection_curve_id: str,
        start_time: float,
        end_time: float,
        accrual_fraction: float,
        quote_weight: float = 1.0,
    ):
        start = _time(start_time, "start_time")
        end = _time(end_time, "end_time")
        if end <= start:
            raise ValueError("end_time must be later than start_time.")
        self.instrument_id = _identifier(instrument_id, "instrument_id")
        self.projection_curve_id = _identifier(projection_curve_id, "projection_curve_id")
        self.start_time = start
        self.end_time = end
        self.accrual_fraction = _positive(accrual_fraction, "accrual_fraction")
        self.quote_weight = _positive(quote_weight, "quote_weight")


class ParSwapBootstrapInstrument(StrictModule):
    """Par fixed rate from explicit discount and projection curves."""

    start_times: Array
    end_times: Array
    payment_times: Array
    accrual_fractions: Array
    valid: Array
    instrument_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    projection_curve_id: str = eqx.field(static=True)
    quote_weight: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        instrument_id: str,
        discount_curve_id: str,
        projection_curve_id: str,
        start_times: ArrayLike,
        end_times: ArrayLike,
        payment_times: ArrayLike,
        accrual_fractions: ArrayLike,
        valid: ArrayLike,
        quote_weight: float = 1.0,
    ):
        start, end, payment, accrual, mask = _schedule_arrays(
            start_times, end_times, payment_times, accrual_fractions, valid
        )
        if not bool(np.any(np.asarray(mask))):
            raise ValueError("A par-swap bootstrap instrument needs an active period.")
        self.start_times = start
        self.end_times = end
        self.payment_times = payment
        self.accrual_fractions = accrual
        self.valid = mask
        self.instrument_id = _identifier(instrument_id, "instrument_id")
        self.discount_curve_id = _identifier(discount_curve_id, "discount_curve_id")
        self.projection_curve_id = _identifier(projection_curve_id, "projection_curve_id")
        self.quote_weight = _positive(quote_weight, "quote_weight")


class BasisSwapBootstrapInstrument(StrictModule):
    """Quoted spread on a receive leg against an explicit pay projection leg."""

    start_times: Array
    end_times: Array
    payment_times: Array
    accrual_fractions: Array
    valid: Array
    instrument_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    receive_projection_curve_id: str = eqx.field(static=True)
    pay_projection_curve_id: str = eqx.field(static=True)
    quote_weight: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        instrument_id: str,
        discount_curve_id: str,
        receive_projection_curve_id: str,
        pay_projection_curve_id: str,
        start_times: ArrayLike,
        end_times: ArrayLike,
        payment_times: ArrayLike,
        accrual_fractions: ArrayLike,
        valid: ArrayLike,
        quote_weight: float = 1.0,
    ):
        start, end, payment, accrual, mask = _schedule_arrays(
            start_times, end_times, payment_times, accrual_fractions, valid
        )
        if not bool(np.any(np.asarray(mask))):
            raise ValueError("A basis-swap bootstrap instrument needs an active period.")
        self.start_times = start
        self.end_times = end
        self.payment_times = payment
        self.accrual_fractions = accrual
        self.valid = mask
        self.instrument_id = _identifier(instrument_id, "instrument_id")
        self.discount_curve_id = _identifier(discount_curve_id, "discount_curve_id")
        self.receive_projection_curve_id = _identifier(
            receive_projection_curve_id, "receive_projection_curve_id"
        )
        self.pay_projection_curve_id = _identifier(
            pay_projection_curve_id, "pay_projection_curve_id"
        )
        self.quote_weight = _positive(quote_weight, "quote_weight")


class SurvivalProbabilityBootstrapInstrument(StrictModule):
    """Direct survival-probability quote equation."""

    instrument_id: str = eqx.field(static=True)
    survival_curve_id: str = eqx.field(static=True)
    maturity: float = eqx.field(static=True)
    quote_weight: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        instrument_id: str,
        survival_curve_id: str,
        maturity: float,
        quote_weight: float = 1.0,
    ):
        self.instrument_id = _identifier(instrument_id, "instrument_id")
        self.survival_curve_id = _identifier(survival_curve_id, "survival_curve_id")
        self.maturity = _positive(maturity, "maturity")
        self.quote_weight = _positive(quote_weight, "quote_weight")


class HazardRateBootstrapInstrument(StrictModule):
    """Instantaneous hazard-rate quote equation."""

    instrument_id: str = eqx.field(static=True)
    survival_curve_id: str = eqx.field(static=True)
    maturity: float = eqx.field(static=True)
    quote_weight: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        instrument_id: str,
        survival_curve_id: str,
        maturity: float,
        quote_weight: float = 1.0,
    ):
        self.instrument_id = _identifier(instrument_id, "instrument_id")
        self.survival_curve_id = _identifier(survival_curve_id, "survival_curve_id")
        self.maturity = _time(maturity, "maturity")
        self.quote_weight = _positive(quote_weight, "quote_weight")


BootstrapInstrument: TypeAlias = (
    DepositBootstrapInstrument
    | ZeroRateBootstrapInstrument
    | ForwardRateBootstrapInstrument
    | ParSwapBootstrapInstrument
    | BasisSwapBootstrapInstrument
    | SurvivalProbabilityBootstrapInstrument
    | HazardRateBootstrapInstrument
)


def _instrument_id(instrument: BootstrapInstrument, /) -> str:
    return instrument.instrument_id


def _quote_weight(instrument: BootstrapInstrument, /) -> float:
    return instrument.quote_weight


def _required_curve_ids(instrument: BootstrapInstrument, /) -> tuple[str, ...]:
    if isinstance(instrument, DepositBootstrapInstrument):
        return (instrument.discount_curve_id,)
    if isinstance(instrument, ZeroRateBootstrapInstrument):
        return (instrument.curve_id,)
    if isinstance(instrument, ForwardRateBootstrapInstrument):
        return (instrument.projection_curve_id,)
    if isinstance(instrument, ParSwapBootstrapInstrument):
        return (instrument.discount_curve_id, instrument.projection_curve_id)
    if isinstance(instrument, BasisSwapBootstrapInstrument):
        return (
            instrument.discount_curve_id,
            instrument.receive_projection_curve_id,
            instrument.pay_projection_curve_id,
        )
    return (instrument.survival_curve_id,)


def _instrument_topology(instrument: BootstrapInstrument, /) -> tuple[Any, ...]:
    common = (
        type(instrument).__name__,
        instrument.instrument_id,
        instrument.quote_weight,
    )
    if isinstance(instrument, DepositBootstrapInstrument):
        return common + (
            instrument.discount_curve_id,
            instrument.start_time,
            instrument.end_time,
            instrument.accrual_fraction,
        )
    if isinstance(instrument, ZeroRateBootstrapInstrument):
        return common + (instrument.curve_id, instrument.maturity)
    if isinstance(instrument, ForwardRateBootstrapInstrument):
        return common + (
            instrument.projection_curve_id,
            instrument.start_time,
            instrument.end_time,
            instrument.accrual_fraction,
        )
    if isinstance(instrument, (ParSwapBootstrapInstrument, BasisSwapBootstrapInstrument)):
        curve_ids = _required_curve_ids(instrument)
        return (
            common
            + curve_ids
            + (
                tuple(np.asarray(instrument.start_times).tolist()),
                tuple(np.asarray(instrument.end_times).tolist()),
                tuple(np.asarray(instrument.payment_times).tolist()),
                tuple(np.asarray(instrument.accrual_fractions).tolist()),
                tuple(np.asarray(instrument.valid).tolist()),
            )
        )
    return common + (instrument.survival_curve_id, instrument.maturity)


def _model_quote(instrument: BootstrapInstrument, curves: CurveSet, /) -> Array:
    if isinstance(instrument, DepositBootstrapInstrument):
        curve = curves.curve(instrument.discount_curve_id)
        return (
            curve.discount_factor(instrument.start_time)
            / curve.discount_factor(instrument.end_time)
            - 1.0
        ) / instrument.accrual_fraction
    if isinstance(instrument, ZeroRateBootstrapInstrument):
        return curves.curve(instrument.curve_id).zero_rate(instrument.maturity)
    if isinstance(instrument, ForwardRateBootstrapInstrument):
        return curves.curve(instrument.projection_curve_id).forward_rate(
            instrument.start_time,
            instrument.end_time,
            accrual_fractions=instrument.accrual_fraction,
        )
    if isinstance(instrument, ParSwapBootstrapInstrument):
        mask = instrument.valid
        safe_payment = jnp.where(mask, instrument.payment_times, 0.0)
        safe_start = jnp.where(mask, instrument.start_times, 0.0)
        safe_end = jnp.where(mask, instrument.end_times, 1.0)
        safe_accrual = jnp.where(mask, instrument.accrual_fractions, 1.0)
        discount = curves.curve(instrument.discount_curve_id).discount_factor(
            safe_payment
        )
        forwards = curves.curve(instrument.projection_curve_id).forward_rate(
            safe_start,
            safe_end,
            accrual_fractions=safe_accrual,
        )
        annuity = jnp.sum(jnp.where(mask, discount * instrument.accrual_fractions, 0.0))
        float_pv = jnp.sum(
            jnp.where(
                mask,
                discount * instrument.accrual_fractions * forwards,
                0.0,
            )
        )
        return float_pv / annuity
    if isinstance(instrument, BasisSwapBootstrapInstrument):
        mask = instrument.valid
        safe_payment = jnp.where(mask, instrument.payment_times, 0.0)
        safe_start = jnp.where(mask, instrument.start_times, 0.0)
        safe_end = jnp.where(mask, instrument.end_times, 1.0)
        safe_accrual = jnp.where(mask, instrument.accrual_fractions, 1.0)
        discount = curves.curve(instrument.discount_curve_id).discount_factor(
            safe_payment
        )
        receive = curves.curve(instrument.receive_projection_curve_id).forward_rate(
            safe_start,
            safe_end,
            accrual_fractions=safe_accrual,
        )
        pay = curves.curve(instrument.pay_projection_curve_id).forward_rate(
            safe_start,
            safe_end,
            accrual_fractions=safe_accrual,
        )
        weighted_accrual = jnp.where(
            mask,
            discount * instrument.accrual_fractions,
            0.0,
        )
        annuity = jnp.sum(weighted_accrual)
        return jnp.sum(weighted_accrual * (pay - receive)) / annuity
    if isinstance(instrument, SurvivalProbabilityBootstrapInstrument):
        return curves.curve(instrument.survival_curve_id).survival_probability(
            instrument.maturity
        )
    if isinstance(instrument, HazardRateBootstrapInstrument):
        return curves.curve(instrument.survival_curve_id).hazard_rate(instrument.maturity)
    raise TypeError("Unsupported bootstrap instrument type.")


class BootstrapSolverPolicy(StrictModule):
    """Numerical termination and independent repricing thresholds."""

    absolute_optimality: float = eqx.field(static=True)
    relative_optimality: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    repricing_tolerance: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    initial_damping: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_optimality: float = 1e-10,
        relative_optimality: float = 1e-10,
        maximum_steps: int = 128,
        repricing_tolerance: float = 1e-9,
        rank_tolerance: float = 1e-10,
        initial_damping: float = 1e-6,
    ):
        self.absolute_optimality = _positive(absolute_optimality, "absolute_optimality")
        self.relative_optimality = _positive(relative_optimality, "relative_optimality")
        steps = int(maximum_steps)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        self.maximum_steps = steps
        self.repricing_tolerance = _positive(repricing_tolerance, "repricing_tolerance")
        self.rank_tolerance = _positive(rank_tolerance, "rank_tolerance")
        self.initial_damping = _positive(initial_damping, "initial_damping")


def _validate_plan_inputs(
    definitions: tuple[CurveDefinition, ...],
    instruments: tuple[BootstrapInstrument, ...],
    initial_node_values: tuple[ArrayLike, ...],
    solver_policy: BootstrapSolverPolicy,
    /,
) -> tuple[
    tuple[CurveDefinition, ...],
    tuple[BootstrapInstrument, ...],
    tuple[Array, ...],
]:
    definitions_ = tuple(definitions)
    instruments_ = tuple(instruments)
    initial_ = tuple(jnp.asarray(values) for values in initial_node_values)
    if not definitions_:
        raise ValueError("A bootstrap plan requires at least one curve definition.")
    if any(not isinstance(value, CurveDefinition) for value in definitions_):
        raise TypeError("definitions must contain CurveDefinition instances.")
    curve_ids = tuple(definition.curve_id for definition in definitions_)
    if len(set(curve_ids)) != len(curve_ids):
        raise ValueError("Bootstrap curve identifiers must be unique.")
    if not instruments_:
        raise ValueError("A bootstrap plan requires at least one instrument.")
    supported = (
        DepositBootstrapInstrument,
        ZeroRateBootstrapInstrument,
        ForwardRateBootstrapInstrument,
        ParSwapBootstrapInstrument,
        BasisSwapBootstrapInstrument,
        SurvivalProbabilityBootstrapInstrument,
        HazardRateBootstrapInstrument,
    )
    if any(not isinstance(value, supported) for value in instruments_):
        raise TypeError("instruments contain an unsupported bootstrap instrument.")
    quote_ids = tuple(_instrument_id(value) for value in instruments_)
    if len(set(quote_ids)) != len(quote_ids):
        raise ValueError("Bootstrap instrument identifiers must be unique.")
    for instrument in instruments_:
        missing = set(_required_curve_ids(instrument)).difference(curve_ids)
        if missing:
            raise ValueError(
                f"Bootstrap instrument {_instrument_id(instrument)!r} references "
                f"unknown curves {sorted(missing)!r}."
            )
    if len(initial_) != len(definitions_):
        raise ValueError("initial_node_values must contain one array per curve.")
    for definition, nodes in zip(definitions_, initial_, strict=True):
        PreparedCurve(definition, nodes)
        if definition.representation is CurveRepresentation.LOG_SURVIVAL:
            concrete = np.asarray(nodes)
            if np.any(np.diff(concrete) >= 0.0):
                raise ValueError(
                    "Bootstrap log-survival initial nodes must be strictly decreasing."
                )
        if definition.representation is CurveRepresentation.HAZARD_RATE:
            concrete = np.asarray(nodes)
            if np.any(concrete <= 0.0):
                raise ValueError(
                    "Bootstrap hazard initial nodes must be strictly positive."
                )
    if not isinstance(solver_policy, BootstrapSolverPolicy):
        raise TypeError("solver_policy must be a BootstrapSolverPolicy.")
    return definitions_, instruments_, initial_


def _inverse_softplus(value: Array, /) -> Array:
    return value + jnp.log(-jnp.expm1(-value))


def _encode_nodes(definition: CurveDefinition, nodes: Array, /) -> Array:
    representation = definition.representation
    if representation is CurveRepresentation.LOG_DISCOUNT:
        return nodes[1:]
    if representation is CurveRepresentation.LOG_SURVIVAL:
        rates = -jnp.diff(nodes) / jnp.diff(definition.grid.times)
        return _inverse_softplus(rates)
    if representation is CurveRepresentation.HAZARD_RATE:
        return _inverse_softplus(nodes)
    return nodes


def _decode_nodes(definition: CurveDefinition, raw: Array, /) -> Array:
    representation = definition.representation
    if representation is CurveRepresentation.LOG_DISCOUNT:
        return jnp.concatenate((jnp.zeros((1,), dtype=raw.dtype), raw))
    if representation is CurveRepresentation.LOG_SURVIVAL:
        increments = jax.nn.softplus(raw) * jnp.diff(definition.grid.times)
        return jnp.concatenate(
            (jnp.zeros((1,), dtype=raw.dtype), -jnp.cumsum(increments))
        )
    if representation is CurveRepresentation.HAZARD_RATE:
        return jax.nn.softplus(raw)
    return raw


def _raw_sizes(definitions: tuple[CurveDefinition, ...], /) -> tuple[int, ...]:
    return tuple(
        definition.grid.node_count - 1
        if definition.representation
        in (CurveRepresentation.LOG_DISCOUNT, CurveRepresentation.LOG_SURVIVAL)
        else definition.grid.node_count
        for definition in definitions
    )


def _split_flat(flat: Array, sizes: tuple[int, ...], /) -> tuple[Array, ...]:
    boundaries = np.cumsum((0,) + sizes)
    return tuple(
        flat[int(start) : int(end)]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def _decode_flat(
    definitions: tuple[CurveDefinition, ...], flat: Array, /
) -> tuple[Array, ...]:
    return tuple(
        _decode_nodes(definition, raw)
        for definition, raw in zip(
            definitions, _split_flat(flat, _raw_sizes(definitions)), strict=True
        )
    )


def _curve_set(
    definitions: tuple[CurveDefinition, ...], nodes: tuple[Array, ...], /
) -> CurveSet:
    return CurveSet(
        tuple(
            PreparedCurve(definition, values)
            for definition, values in zip(definitions, nodes, strict=True)
        )
    )


def _model_quotes(
    instruments: tuple[BootstrapInstrument, ...], curves: CurveSet, /
) -> Array:
    return jnp.stack(
        tuple(_model_quote(instrument, curves) for instrument in instruments)
    )


class AbstractCurveBootstrapPlan(StrictModule):
    """Shared typed bootstrap execution without storing an opaque residual callable."""

    definitions: AbstractAttribute[tuple[CurveDefinition, ...]]
    instruments: AbstractAttribute[tuple[BootstrapInstrument, ...]]
    initial_node_values: AbstractAttribute[tuple[Array, ...]]
    solver_policy: AbstractAttribute[BootstrapSolverPolicy]
    quote_ids: AbstractAttribute[tuple[str, ...]]

    @property
    def topology_key(self) -> tuple[Any, ...]:
        return (
            tuple(definition.topology_key for definition in self.definitions),
            tuple(_instrument_topology(instrument) for instrument in self.instruments),
            self.quote_ids,
            _raw_sizes(self.definitions),
        )

    def model_quotes(self, node_values: tuple[Array, ...], /) -> Array:
        if len(node_values) != len(self.definitions):
            raise ValueError("node_values must contain one array per curve.")
        return _model_quotes(
            self.instruments, _curve_set(self.definitions, tuple(node_values))
        )

    def residual(self, raw_parameters: ArrayLike, quotes: ArrayLike, /) -> Array:
        raw = jnp.asarray(raw_parameters)
        quotes_ = jnp.asarray(quotes)
        if raw.shape != (sum(_raw_sizes(self.definitions)),):
            raise ValueError("raw_parameters do not match the bootstrap node layout.")
        if quotes_.shape != (len(self.instruments),):
            raise ValueError("quotes do not match the bootstrap instrument layout.")
        curves = _curve_set(self.definitions, _decode_flat(self.definitions, raw))
        model = _model_quotes(self.instruments, curves)
        weights = jnp.asarray(tuple(_quote_weight(value) for value in self.instruments))
        return weights * (model - quotes_)

    def solve(self, quotes: ArrayLike, /) -> CurveBootstrapResult:
        initial_raw = jnp.concatenate(
            tuple(
                _encode_nodes(definition, nodes)
                for definition, nodes in zip(
                    self.definitions, self.initial_node_values, strict=True
                )
            )
        )
        return self._solve_from_raw(quotes, initial_raw)

    def _solve_from_raw(
        self, quotes: ArrayLike, initial_raw: ArrayLike, /
    ) -> CurveBootstrapResult:
        quotes_ = jnp.asarray(quotes)
        if quotes_.shape != (len(self.quote_ids),):
            raise ValueError("quotes must match the plan's ordered quote layout.")
        if jnp.issubdtype(quotes_.dtype, jnp.complexfloating):
            raise TypeError("Bootstrap quotes must be real-valued.")
        quotes_ = eqx.error_if(
            quotes_, jnp.any(~jnp.isfinite(quotes_)), "Bootstrap quotes must be finite."
        )
        initial = jnp.asarray(initial_raw)
        if initial.shape != (sum(_raw_sizes(self.definitions)),):
            raise ValueError("initial_raw must preserve the plan's free-node layout.")
        policy = self.solver_policy
        optimization = least_squares(
            lambda parameters, observations: self.residual(parameters, observations),
            initial,
            method=LevenbergMarquardt(initial_damping=policy.initial_damping),
            termination=OptimizationTermination(
                absolute_optimality=policy.absolute_optimality,
                relative_optimality=policy.relative_optimality,
                absolute_step=policy.absolute_optimality,
                relative_step=policy.relative_optimality,
                maximum_steps=policy.maximum_steps,
            ),
            args=quotes_,
        )
        fitted_raw = optimization.parameters
        fitted_nodes = _decode_flat(self.definitions, fitted_raw)
        bare_curves = _curve_set(self.definitions, fitted_nodes)
        repriced = _model_quotes(self.instruments, bare_curves)
        errors = repriced - quotes_
        weighted_jacobian = jax.jacfwd(lambda raw: self.residual(raw, quotes_))(
            fitted_raw
        )
        factorization = factorize(
            DenseLinearOperator(weighted_jacobian),
            FactorizationPolicy(
                "svd",
                rank=RankPolicy(relative_cutoff=policy.rank_tolerance),
            ),
        )
        singular_values = factorization.singular_values()
        free_count = int(fitted_raw.shape[0])
        independent = (len(self.instruments) >= free_count) & (
            jnp.sum(singular_values > policy.rank_tolerance) == free_count
        )
        weights = jnp.asarray(tuple(_quote_weight(value) for value in self.instruments))
        raw_quote_jacobian = (
            factorization.materialize_pseudoinverse().value * weights[None, :]
        )

        def actual_flat(raw):
            return jnp.concatenate(_decode_flat(self.definitions, raw))

        actual_raw_jacobian = jax.jacfwd(actual_flat)(fitted_raw)
        actual_quote_jacobian = actual_raw_jacobian @ raw_quote_jacobian
        node_sizes = tuple(definition.grid.node_count for definition in self.definitions)
        quote_blocks = _split_flat(actual_quote_jacobian, node_sizes)
        calibrated_curves = CurveSet(
            tuple(
                PreparedCurve(
                    definition,
                    nodes,
                    node_quote_jacobian=quote_jacobian,
                    quote_ids=self.quote_ids,
                )
                for definition, nodes, quote_jacobian in zip(
                    self.definitions, fitted_nodes, quote_blocks, strict=True
                )
            )
        )
        finite = (
            jnp.all(jnp.isfinite(fitted_raw))
            & jnp.all(jnp.isfinite(repriced))
            & jnp.all(jnp.isfinite(actual_quote_jacobian))
        )
        max_error = jnp.max(jnp.abs(errors))
        repriced_independently = (
            finite & independent & (max_error <= policy.repricing_tolerance)
        )
        successful = optimization.successful & repriced_independently
        return CurveBootstrapResult(
            plan=self,
            curves=calibrated_curves,
            quotes=quotes_,
            model_quotes=repriced,
            repricing_errors=errors,
            fitted_raw_parameters=fitted_raw,
            optimizer_status=optimization.status,
            iterations=optimization.diagnostics.iterations,
            jacobian_rank=jnp.sum(
                singular_values > policy.rank_tolerance, dtype=jnp.int32
            ),
            independent=independent,
            repriced=repriced_independently,
            successful=successful,
        )

    def prepare_replay(self) -> CurveBootstrapReplay:
        initial_raw = jnp.concatenate(
            tuple(
                _encode_nodes(definition, nodes)
                for definition, nodes in zip(
                    self.definitions, self.initial_node_values, strict=True
                )
            )
        )
        return CurveBootstrapReplay(self, initial_raw)


class SingleCurveBootstrapPlan(AbstractCurveBootstrapPlan):
    """Bootstrap plan whose equations identify exactly one curve."""

    definitions: tuple[CurveDefinition, ...]
    instruments: tuple[BootstrapInstrument, ...]
    initial_node_values: tuple[Array, ...]
    solver_policy: BootstrapSolverPolicy
    quote_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        definition: CurveDefinition,
        instruments: tuple[BootstrapInstrument, ...],
        initial_node_values: ArrayLike,
        /,
        *,
        solver_policy: BootstrapSolverPolicy | None = None,
    ):
        policy = BootstrapSolverPolicy() if solver_policy is None else solver_policy
        definitions, instruments_, initial = _validate_plan_inputs(
            (definition,), instruments, (initial_node_values,), policy
        )
        self.definitions = definitions
        self.instruments = instruments_
        self.initial_node_values = initial
        self.solver_policy = policy
        self.quote_ids = tuple(_instrument_id(value) for value in instruments_)


class MultiCurveBootstrapPlan(AbstractCurveBootstrapPlan):
    """Joint bootstrap plan for two or more explicitly identified curves."""

    definitions: tuple[CurveDefinition, ...]
    instruments: tuple[BootstrapInstrument, ...]
    initial_node_values: tuple[Array, ...]
    solver_policy: BootstrapSolverPolicy
    quote_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        definitions: tuple[CurveDefinition, ...],
        instruments: tuple[BootstrapInstrument, ...],
        initial_node_values: tuple[ArrayLike, ...],
        /,
        *,
        solver_policy: BootstrapSolverPolicy | None = None,
    ):
        if len(definitions) < 2:
            raise ValueError("MultiCurveBootstrapPlan requires at least two curves.")
        policy = BootstrapSolverPolicy() if solver_policy is None else solver_policy
        definitions_, instruments_, initial = _validate_plan_inputs(
            definitions, instruments, initial_node_values, policy
        )
        self.definitions = definitions_
        self.instruments = instruments_
        self.initial_node_values = initial
        self.solver_policy = policy
        self.quote_ids = tuple(_instrument_id(value) for value in instruments_)


class CurveBootstrapResult(StrictModule):
    """Fitted curves plus independent rank and repricing evidence."""

    plan: AbstractCurveBootstrapPlan
    curves: CurveSet
    quotes: Array
    model_quotes: Array
    repricing_errors: Array
    fitted_raw_parameters: Array
    optimizer_status: Array
    iterations: Array
    jacobian_rank: Array
    independent: Array
    repriced: Array
    successful: Array

    def __init__(
        self,
        *,
        plan: AbstractCurveBootstrapPlan,
        curves: CurveSet,
        quotes: ArrayLike,
        model_quotes: ArrayLike,
        repricing_errors: ArrayLike,
        fitted_raw_parameters: ArrayLike,
        optimizer_status: ArrayLike,
        iterations: ArrayLike,
        jacobian_rank: ArrayLike,
        independent: ArrayLike,
        repriced: ArrayLike,
        successful: ArrayLike,
    ):
        self.plan = plan
        self.curves = curves
        self.quotes = jnp.asarray(quotes)
        self.model_quotes = jnp.asarray(model_quotes)
        self.repricing_errors = jnp.asarray(repricing_errors)
        self.fitted_raw_parameters = jnp.asarray(fitted_raw_parameters)
        self.optimizer_status = jnp.asarray(optimizer_status, dtype=jnp.int32)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.jacobian_rank = jnp.asarray(jacobian_rank, dtype=jnp.int32)
        self.independent = jnp.asarray(independent, dtype=bool)
        self.repriced = jnp.asarray(repriced, dtype=bool)
        self.successful = jnp.asarray(successful, dtype=bool)

    @property
    def maximum_absolute_repricing_error(self) -> Array:
        return jnp.max(jnp.abs(self.repricing_errors))

    def require_success(self) -> CurveSet:
        successful = np.asarray(self.successful)
        if successful.ndim != 0:
            raise ValueError("Bootstrap success evidence must be scalar.")
        if not bool(successful):
            raise RuntimeError(
                "Curve bootstrap did not independently identify and reprice its instruments."
            )
        return self.curves

    def replay(self) -> CurveBootstrapReplay:
        return CurveBootstrapReplay(self.plan, self.fitted_raw_parameters)


class CurveBootstrapReplay(StrictModule):
    """Warm-started numeric replay bound to one exact topology and quote layout."""

    plan: AbstractCurveBootstrapPlan
    initial_raw_parameters: Array
    topology_key: tuple[Any, ...] = eqx.field(static=True)
    quote_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        plan: AbstractCurveBootstrapPlan,
        initial_raw_parameters: ArrayLike,
        /,
    ):
        if not isinstance(plan, AbstractCurveBootstrapPlan):
            raise TypeError("plan must be an AbstractCurveBootstrapPlan.")
        initial = jnp.asarray(initial_raw_parameters)
        if initial.shape != (sum(_raw_sizes(plan.definitions)),):
            raise ValueError("initial_raw_parameters do not match the plan topology.")
        self.plan = plan
        self.initial_raw_parameters = initial
        self.topology_key = plan.topology_key
        self.quote_ids = plan.quote_ids

    def replay(self, quotes: ArrayLike, /) -> CurveBootstrapResult:
        return self.plan._solve_from_raw(quotes, self.initial_raw_parameters)

    def refresh(
        self,
        plan: AbstractCurveBootstrapPlan,
        quotes: ArrayLike,
        /,
    ) -> CurveBootstrapResult:
        if not isinstance(plan, AbstractCurveBootstrapPlan):
            raise TypeError("plan must be an AbstractCurveBootstrapPlan.")
        if plan.topology_key != self.topology_key:
            raise ValueError("Numeric replay must preserve bootstrap topology.")
        if plan.quote_ids != self.quote_ids:
            raise ValueError("Numeric replay must preserve the ordered quote layout.")
        return plan._solve_from_raw(quotes, self.initial_raw_parameters)


__all__ = [
    "AbstractCurveBootstrapPlan",
    "BasisSwapBootstrapInstrument",
    "BootstrapInstrument",
    "BootstrapSolverPolicy",
    "CurveBootstrapReplay",
    "CurveBootstrapResult",
    "DepositBootstrapInstrument",
    "ForwardRateBootstrapInstrument",
    "HazardRateBootstrapInstrument",
    "MultiCurveBootstrapPlan",
    "ParSwapBootstrapInstrument",
    "SingleCurveBootstrapPlan",
    "SurvivalProbabilityBootstrapInstrument",
    "ZeroRateBootstrapInstrument",
]

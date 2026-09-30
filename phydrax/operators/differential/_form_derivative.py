"""Derivative-rule composition for smooth form evaluators.

Analytic coordinate rules augment, rather than replace, native parameter AD.
Operand DomainFunctions remain explicit PyTree arguments of the custom JVP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from phydrax.domain import (
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
    DomainFunction,
)
from phydrax.domain._derivative import DerivativeRuleProvider

from ..._strict import StrictModule
from ._domain_ops import _coord_axis_position, _factor_and_dim, grad, partial, partial_n


if TYPE_CHECKING:
    from ...nn._keys import EvalKey

type _DerivativeOptions = tuple[DerivativeMode, DerivativeBackend, DerivativeBasis, bool]
type _PartialRequest = tuple[str, int | None, int, _DerivativeOptions]
_DEFAULT_OPTIONS: _DerivativeOptions = ("forward", "ad", "poly", False)
_OPTIONS_KEY = "_form_derivative_options"


class _FormDerivativeProvider(DerivativeRuleProvider):
    """Compose coordinate rules from the evaluator's live numerical operands."""

    def derivative_rule_for(self, function: DomainFunction, /) -> DerivativeRule:
        return _CompositionDerivativeRule(function, ())


def _raw_function_value(
    function: DomainFunction,
    args: tuple[Any, ...],
    key: EvalKey,
    kwargs: dict[str, Any],
    options: _DerivativeOptions,
) -> Array:
    if isinstance(function.func, _FormDerivativeProvider):
        return jnp.asarray(
            function.func(*args, key=key, **kwargs, **{_OPTIONS_KEY: options})
        )
    return jnp.asarray(function.func(*args, key=key, **kwargs))


@eqx.filter_custom_jvp
def _rule_function_value(
    function: DomainFunction,
    args: tuple[Any, ...],
    key: EvalKey,
    kwargs: dict[str, Any],
    options: _DerivativeOptions,
) -> Array:
    return _raw_function_value(function, args, key, kwargs, options)


def _coordinate_tangent_term(
    function: DomainFunction,
    args: tuple[Any, ...],
    key: EvalKey,
    kwargs: dict[str, Any],
    options: _DerivativeOptions,
    position: int,
    axis: int | None,
    tangent: Array,
) -> Array:
    derivative = _native_partial_function(
        function, ((function.deps[position], axis, 1, options),)
    )
    values = _rule_function_value(derivative, args, key, kwargs, options)
    if isinstance(args[position], tuple):
        coordinate_axis = _coord_axis_position(
            args, arg_index=position, coord_index=0 if axis is None else axis
        )
        shape = [1] * values.ndim
        shape[coordinate_axis] = tangent.shape[0]
        tangent = tangent.reshape(tuple(shape))
    else:
        tangent = tangent.reshape(
            (*tangent.shape, *(1 for _ in range(values.ndim - tangent.ndim)))
        )
    return values * tangent


@_rule_function_value.def_jvp
def _rule_function_value_jvp(
    primals: tuple[
        DomainFunction, tuple[Any, ...], EvalKey, dict[str, Any], _DerivativeOptions
    ],
    tangents: tuple[
        DomainFunction, tuple[Any, ...], EvalKey, dict[str, Any], tuple[None, ...]
    ],
) -> tuple[Array, Array]:
    function, args, key, kwargs, options = primals
    function_tangent, args_tangent, key_tangent, kwargs_tangent, options_tangent = (
        tangents
    )
    # Parameter tangents have native AD; only coordinate tangents use analytic rules.
    value, parameter_tangent = eqx.filter_jvp(
        _raw_function_value,
        primals,
        (
            function_tangent,
            jax.tree.map(lambda _: None, args),
            key_tangent,
            kwargs_tangent,
            options_tangent,
        ),
    )
    tangent = jnp.zeros_like(value) if parameter_tangent is None else parameter_tangent
    for position, label in enumerate(function.deps):
        coordinate_tangent = args_tangent[position]
        if coordinate_tangent is None:
            continue
        factor, dimension = _factor_and_dim(function, label)
        if isinstance(args[position], tuple):
            for axis, term in enumerate(coordinate_tangent):
                if term is not None:
                    tangent = tangent + _coordinate_tangent_term(
                        function, args, key, kwargs, options, position, axis, term
                    )
        elif factor.kind == "scalar":
            tangent = tangent + _coordinate_tangent_term(
                function, args, key, kwargs, options, position, None, coordinate_tangent
            )
        else:
            for axis in range(dimension):
                tangent = tangent + _coordinate_tangent_term(
                    function,
                    args,
                    key,
                    kwargs,
                    options,
                    position,
                    axis,
                    coordinate_tangent[..., axis],
                )
    return value, tangent


def _function_value(
    function: DomainFunction,
    args: tuple[Any, ...],
    /,
    *,
    key: EvalKey,
    kwargs: dict[str, Any],
) -> Array:
    options: _DerivativeOptions = kwargs.get(_OPTIONS_KEY, _DEFAULT_OPTIONS)
    raw_kwargs = {name: value for name, value in kwargs.items() if name != _OPTIONS_KEY}
    if function.derivative_rule is None:
        return _raw_function_value(function, args, key, raw_kwargs, options)
    return _rule_function_value(function, args, key, raw_kwargs, options)


def _append_partial(
    requests: tuple[_PartialRequest, ...], request: _PartialRequest, /
) -> tuple[_PartialRequest, ...]:
    if requests and requests[-1][:2] == request[:2]:
        last = requests[-1]
        if last[3][1] in ("ad", "jet") and request[3][1] in ("ad", "jet"):
            # AD mode and analytic backend select evaluation, not a new derivative
            # identity. Request the complete order from the original rule.
            return (*requests[:-1], (last[0], last[1], last[2] + request[2], request[3]))
    return (*requests, request)


@dataclass(frozen=True, slots=True, eq=False)
class _NativePartialDerivativeRule(DerivativeRule):
    function: DomainFunction
    requests: tuple[_PartialRequest, ...]

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        if backend not in ("ad", "jet"):
            return None
        request = (var, axis, order, (mode, backend, basis, periodic))
        return _native_partial_function(
            self.function, _append_partial(self.requests, request)
        )


@final
class _NativePartialCallable(StrictModule, _FormDerivativeProvider):
    source: DomainFunction
    requests: tuple[_PartialRequest, ...] = eqx.field(static=True)

    def __init__(
        self, source: DomainFunction, requests: tuple[_PartialRequest, ...], /
    ) -> None:
        self.source = source
        self.requests = requests

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        derivative = self.source
        for var, axis, order, options in self.requests:
            mode, backend, basis, periodic = options
            rule = derivative.derivative_rule
            native = (
                None
                if rule is None
                else rule.derive(
                    var=var,
                    axis=axis,
                    order=order,
                    mode=mode,
                    backend=backend,
                    basis=basis,
                    periodic=periodic,
                )
            )
            derivative = (
                native
                if native is not None
                else partial_n(
                    derivative,
                    var=var,
                    axis=axis,
                    order=order,
                    mode=mode,
                    backend=backend,
                    basis=basis,
                    periodic=periodic,
                )
            )
        return _function_value(derivative, args, key=key, kwargs=kwargs)

    def derivative_rule_for(self, function: DomainFunction, /) -> DerivativeRule:
        del function
        return _NativePartialDerivativeRule(self.source, self.requests)


def _native_partial_function(
    source: DomainFunction, requests: tuple[_PartialRequest, ...], /
) -> DomainFunction:
    return DomainFunction(
        domain=source.domain,
        deps=source.deps,
        metadata=source.metadata,
        func=_NativePartialCallable(source, requests),
    )


@dataclass(frozen=True, slots=True, eq=False)
class _CompositionDerivativeRule(DerivativeRule):
    function: DomainFunction
    requests: tuple[_PartialRequest, ...]

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        if backend not in ("ad", "jet"):
            return None
        request = (var, axis, order, (mode, backend, basis, periodic))
        return DomainFunction(
            domain=self.function.domain,
            deps=self.function.deps,
            metadata=self.function.metadata,
            func=_CompositionPartialCallable(
                self.function, _append_partial(self.requests, request)
            ),
        )


@final
class _CompositionPartialCallable(StrictModule, _FormDerivativeProvider):
    source: DomainFunction
    requests: tuple[_PartialRequest, ...] = eqx.field(static=True)

    def __init__(
        self, source: DomainFunction, requests: tuple[_PartialRequest, ...], /
    ) -> None:
        self.source = source
        self.requests = requests

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        derivative = self.source
        for var, axis, order, options in self.requests:
            # Bypass the operation's own rule; its operand custom JVPs use native rules.
            derivative = DomainFunction(
                domain=derivative.domain,
                deps=derivative.deps,
                metadata=derivative.metadata,
                func=_RuleContextCallable(derivative, options),
            )
            for _ in range(order):
                derivative = partial(
                    derivative, var=var, axis=axis, mode=options[0], ad_engine="jvp"
                )
        return jnp.asarray(
            derivative.func(
                *args,
                key=key,
                **{name: value for name, value in kwargs.items() if name != _OPTIONS_KEY},
            )
        )

    def derivative_rule_for(self, function: DomainFunction, /) -> DerivativeRule:
        del function
        return _CompositionDerivativeRule(self.source, self.requests)


@final
class _RuleContextCallable(StrictModule):
    source: DomainFunction
    options: _DerivativeOptions = eqx.field(static=True)

    def __init__(self, source: DomainFunction, options: _DerivativeOptions, /) -> None:
        self.source = source
        self.options = options

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        raw_kwargs = {
            name: value for name, value in kwargs.items() if name != _OPTIONS_KEY
        }
        return _raw_function_value(self.source, args, key, raw_kwargs, self.options)


@final
class _GradientCallable(StrictModule, _FormDerivativeProvider):
    source: DomainFunction
    var: str = eqx.field(static=True)
    options: _DerivativeOptions = eqx.field(static=True)

    def __init__(
        self, source: DomainFunction, var: str, options: _DerivativeOptions, /
    ) -> None:
        self.source = source
        self.var = var
        self.options = options

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        mode, backend, basis, periodic = self.options
        if backend in ("ad", "jet") and self.source.derivative_rule is not None:
            factor, dimension = _factor_and_dim(self.source, self.var)
            partials = tuple(
                _function_value(
                    _native_partial_function(
                        self.source,
                        (
                            (
                                self.var,
                                None if factor.kind == "scalar" else axis,
                                1,
                                self.options,
                            ),
                        ),
                    ),
                    args,
                    key=key,
                    kwargs=kwargs,
                )
                for axis in range(dimension)
            )
            return jnp.stack(partials, axis=-1)
        derivative = grad(
            self.source,
            var=self.var,
            mode=mode,
            backend=backend,
            basis=basis,
            periodic=periodic,
        )
        return _function_value(derivative, args, key=key, kwargs=kwargs)


def _gradient_function(
    source: DomainFunction, var: str, options: _DerivativeOptions, /
) -> DomainFunction:
    return DomainFunction(
        domain=source.domain,
        deps=source.deps,
        metadata=source.metadata,
        func=_GradientCallable(source, var, options),
    )

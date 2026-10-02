from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._strict import StrictModule
from phydrax.domain import (
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
    Domain,
    DomainFunction,
    Interval1d,
    TimeInterval,
)
from phydrax.domain._derivative import DerivativeRuleProvider
from phydrax.operators.differential import (
    DerivativeStep,
    laplacian,
    partial_n,
    trace_derivative_requests,
)
from phydrax.typing import PRNGKey


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


@final
class _MixedField(StrictModule):
    scale: Array
    complex_output: bool = eqx.field(static=True)

    def __init__(self, scale: Array, *, complex_output: bool = False) -> None:
        self.scale = scale
        self.complex_output = complex_output

    def __call__(self, x: Array, t: Array, *, key: PRNGKey | None = None) -> Array:
        del key
        result = self.scale * x[0] ** 2 * t
        if self.complex_output:
            result = result.astype(jnp.complex128) * jnp.asarray(
                1 + 2j, dtype=jnp.complex128
            )
        return result


@eqx.filter_jit
def _evaluate(field: DomainFunction, x: Array, t: Array) -> Array:
    return field.func(x, t)


@pytest.mark.parametrize("complex_output", [False, True], ids=["real", "complex"])
def test_whole_mixed_path_reuses_live_model_and_preserves_parameter_gradient(
    complex_output: bool,
) -> None:
    domain = Interval1d(-1.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = _MixedField(
        jnp.asarray(3.0, dtype=jnp.float64), complex_output=complex_output
    )
    field = domain.Function("x", "t")(model)
    derivative = partial_n(
        partial_n(field, var="x", axis=0, order=2, backend="jet"),
        var="t",
        order=1,
        backend="jet",
    )
    point = jnp.asarray([0.2], dtype=jnp.float64)
    time = jnp.asarray(0.4, dtype=jnp.float64)
    factor = 1 + 2j if complex_output else 1.0
    np.testing.assert_allclose(
        _evaluate(derivative, point, time), 6.0 * factor, rtol=1e-10
    )

    def loss(current: _MixedField) -> Array:
        rebound = eqx.tree_at(lambda item: item.func.source.func, derivative, current)
        return jnp.real(_evaluate(rebound, point + 0.1, time + 0.2))

    gradient = eqx.filter_grad(loss)(model)
    np.testing.assert_allclose(gradient.scale, 2.0, rtol=1e-10)
    changed = _MixedField(
        jnp.asarray(5.0, dtype=jnp.float64), complex_output=complex_output
    )
    rebound = eqx.tree_at(lambda item: item.func.source.func, derivative, changed)
    np.testing.assert_allclose(_evaluate(rebound, point, time), 10.0 * factor, rtol=1e-10)


def test_trace_preserves_partial_laplacian_interleaving() -> None:
    domain = Interval1d(-1.0, 1.0)
    field = domain.Function("x")(lambda x: x[0] ** 4)

    def laplacian_then_partial(fields: Mapping[str, DomainFunction]) -> DomainFunction:
        return partial_n(laplacian(fields["u"], var="x"), var="x", axis=0, order=1)

    def partial_then_laplacian(fields: Mapping[str, DomainFunction]) -> DomainFunction:
        return laplacian(partial_n(fields["u"], var="x", axis=0, order=1), var="x")

    first = tuple(
        request
        for request in trace_derivative_requests(laplacian_then_partial, {"u": field})
        if request.order == 3
    )
    second = tuple(
        request
        for request in trace_derivative_requests(partial_then_laplacian, {"u": field})
        if request.order == 3
    )
    assert first[0].steps == (
        DerivativeStep("laplacian", "x", order=2),
        DerivativeStep("partial", "x", 0),
    )
    assert second[0].steps == tuple(reversed(first[0].steps))
    assert first[0] != second[0]
    point = jnp.asarray([0.2], dtype=jnp.float64)
    np.testing.assert_allclose(laplacian_then_partial({"u": field}).func(point), 4.8)
    np.testing.assert_allclose(partial_then_laplacian({"u": field}).func(point), 4.8)


@final
class _OwnedQuartic(StrictModule, DerivativeRuleProvider):
    scale: Array

    def __init__(self, scale: Array) -> None:
        self.scale = scale

    def __call__(self, x: Array, *, key: PRNGKey | None = None) -> Array:
        import jax

        del key
        # The native rule owns coordinate differentiation; parameter AD is live.
        return self.scale * jax.lax.stop_gradient(x[0]) ** 4

    def derivative_rule_for(self, function: DomainFunction, /) -> DerivativeRule:
        return _QuarticRule(self, function.domain)


@final
class _QuarticDerivative(StrictModule):
    source: _OwnedQuartic
    order: int = eqx.field(static=True)

    def __init__(self, source: _OwnedQuartic, order: int) -> None:
        self.source = source
        self.order = order

    def __call__(self, x: Array, *, key: PRNGKey | None = None) -> Array:
        import math

        del key
        coefficient = math.factorial(4) // math.factorial(4 - self.order)
        return (
            self.source.scale
            * jnp.asarray(coefficient, dtype=x.dtype)
            * x[0] ** (4 - self.order)
        )


@final
@dataclass(frozen=True, slots=True, eq=False)
class _QuarticRule(DerivativeRule):
    source: _OwnedQuartic
    domain: Domain

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
        del var, axis, order, mode, backend, basis, periodic
        return None

    def derive_path(
        self,
        steps: tuple[DerivativeStep, ...],
        /,
        *,
        mode: DerivativeMode,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, basis, periodic
        if any(step.kind != "partial" or step.variable != "x" for step in steps):
            return None
        order = sum(step.order for step in steps)
        if order > 4:
            return None
        return self.domain.Function("x")(_QuarticDerivative(self.source, order))


def test_complete_native_path_owns_coordinate_derivative_and_parameter_ad() -> None:
    domain = Interval1d(-1.0, 1.0)
    point = jnp.asarray([0.5], dtype=jnp.float64)

    def loss(scale: Array) -> Array:
        field = domain.Function("x")(_OwnedQuartic(scale))
        return partial_n(field, var="x", axis=0, order=2, backend="jet").func(point)

    import jax

    scale = jnp.asarray(3.0, dtype=jnp.float64)
    np.testing.assert_allclose(jax.jit(loss)(scale), 9.0, rtol=1e-12)
    np.testing.assert_allclose(jax.grad(loss)(scale), 3.0, rtol=1e-12)


def test_domain_taylor_refuses_nonfinite_result_at_consumer_boundary() -> None:
    domain = Interval1d(-1.0, 1.0) @ TimeInterval(0.0, 1.0)
    field = domain.Function("x", "t")(
        _MixedField(jnp.asarray(jnp.inf, dtype=jnp.float64))
    )
    derivative = partial_n(field, var="x", axis=0, order=2, backend="jet")
    with pytest.raises(eqx.EquinoxRuntimeError, match="Taylor coordinate contraction"):
        _evaluate(
            derivative,
            jnp.asarray([0.2], dtype=jnp.float64),
            jnp.asarray(0.4, dtype=jnp.float64),
        )


def test_nondifferentiated_pytree_and_integer_data_remain_live() -> None:
    import jax

    from phydrax.domain import TrajectoryDatasetDomain

    domain = TrajectoryDatasetDomain(
        {"a": jnp.asarray([[3]], dtype=jnp.int32)},
        jnp.asarray([3], dtype=jnp.int32),
        dt=jnp.asarray(0.1, dtype=jnp.float64),
    )

    def function(data: Mapping[str, Array], t: Array) -> Array:
        return data["a"][0].astype(t.dtype) * t**2

    field = domain.Function("data", "t")(function)
    derivative = partial_n(field, var="t", order=2, backend="jet")

    @jax.jit
    def evaluate(data: Array, t: Array) -> Array:
        return derivative.func({"a": data}, t)

    time = jnp.asarray(0.3, dtype=jnp.float64)
    np.testing.assert_allclose(evaluate(jnp.asarray([3], dtype=jnp.int32), time), 6.0)
    np.testing.assert_allclose(evaluate(jnp.asarray([5], dtype=jnp.int32), time), 10.0)
    np.testing.assert_allclose(
        jax.grad(lambda coefficient: evaluate(coefficient[None], time))(
            jnp.asarray(3.0, dtype=jnp.float64)
        ),
        2.0,
    )


def test_zero_laplacian_preserves_vector_coordinate_identity() -> None:
    from phydrax.domain import HyperRectangle
    from phydrax.operators.differential._domain_ops import _try_derivative_path

    domain = HyperRectangle(
        jnp.asarray([-1.0, -1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        label="x",
    )
    field = domain.Function()(jnp.asarray(3.0, dtype=jnp.float64))
    derivative = _try_derivative_path(
        field, (DerivativeStep("laplacian", "x", order=2, backend="jet"),)
    )
    if derivative is None:
        raise RuntimeError("An independent coordinate derivative must be owned as zero.")
    np.testing.assert_allclose(derivative.func(), 0.0)

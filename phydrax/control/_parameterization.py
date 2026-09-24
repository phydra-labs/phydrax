#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from math import prod
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._differentiation import ComponentAuthority
from .._interpolation import (
    bspline_evaluate,
    BSplineGrid,
    BSplineGridTransfer,
    ProjectionMethod,
)
from .._model import (
    AbstractArrayModel,
    AbstractComponentSlot,
    bind_component,
    ComponentContract,
)
from .._strict import StrictModule
from .._trainable import fixed_field
from ..dynamics import TimeGrid
from ._problem import _identifier, _shape


def _coefficient_array(
    coefficients: ArrayLike,
    case_shape: tuple[int, ...],
    parameter_shape: tuple[int, ...],
    /,
) -> Array:
    values = jnp.asarray(coefficients)
    expected = case_shape + parameter_shape
    if tuple(values.shape) != expected:
        raise ValueError(
            f"Control coefficients must have shape {expected}; got {values.shape}."
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype("float64")
    return values


def _case_shape(value: Sequence[int], /) -> tuple[int, ...]:
    cases = tuple(value)
    if any(size <= 0 for size in cases):
        raise ValueError("Control coefficient case dimensions must be positive.")
    return cases


def _query(value: ArrayLike, /) -> Array:
    query = jnp.asarray(value)
    if jnp.issubdtype(query.dtype, jnp.complexfloating):
        raise TypeError("Control evaluation times must be real-valued.")
    return query.astype(jnp.result_type(query, jnp.float64))


class AbstractControlParameterization(AbstractComponentSlot):
    """Fixed-shape map from coefficients to physical controls.

    The base is the neutral `DECISION` slot of control problems: a
    parameterization decides the applied control, and every rollout evaluates
    the controlled dynamics on the returned control. Analytic parameterizations
    carry no trainable arrays of their own; their coefficients are decision
    variables supplied at evaluation. A learned policy (`NeuralFeedbackPolicy`)
    holds its model as a dynamic child.
    """

    component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.DECISION
    slot_semantic_id: ClassVar[str] = "phydrax.control.parameterization"

    control_shape: tuple[int, ...] = eqx.field(static=True)
    time_grid: eqx.AbstractVar[TimeGrid | None]
    parameter_shape: tuple[int, ...] = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @abstractmethod
    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        """Evaluate with shape ``case_shape + time.shape + control_shape``."""
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        """Sample an open-loop parameterization at shared physical times."""
        raise NotImplementedError


def _validate_parameterization_grid(
    parameterization: AbstractControlParameterization,
    time_grid: TimeGrid,
    /,
) -> AbstractControlParameterization:
    bound_grid = parameterization.time_grid
    if bound_grid is None:
        return parameterization
    if (
        bound_grid.time_id != time_grid.time_id
        or bound_grid.times.shape != time_grid.times.shape
    ):
        raise ValueError(
            "Grid-bound control parameterization must use the exact problem time grid."
        )
    checked_times = eqx.error_if(
        bound_grid.times,
        jnp.any(bound_grid.times != time_grid.times),
        "Grid-bound control parameterization must use the exact problem time grid.",
    )
    return eqx.tree_at(
        lambda value: value.time_grid.times,
        parameterization,
        checked_times,
    )


class PiecewiseConstantControlParameterization(AbstractControlParameterization):
    """Left-endpoint-held interval controls on a fixed physical time grid."""

    time_grid: TimeGrid

    def __init__(
        self,
        time_grid: TimeGrid,
        control_shape: Sequence[int],
        /,
        *,
        parameterization_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        shape = _shape(control_shape, "control_shape")
        self.time_grid = time_grid
        self.control_shape = shape
        self.parameter_shape = (time_grid.num_steps,) + shape
        self.parameterization_id = _identifier(parameterization_id, "parameterization_id")
        self.approximation_id = "control:piecewise-constant:left-endpoint-hold"

    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        del state
        cases = _case_shape(case_shape)
        values = _coefficient_array(coefficients, cases, self.parameter_shape)
        query = _query(time)
        query = eqx.error_if(
            query,
            jnp.any(~jnp.isfinite(query))
            | jnp.any(query < self.time_grid.t0)
            | jnp.any(query > self.time_grid.t1),
            "Piecewise-constant control time lies outside its physical grid.",
        )
        indices = jnp.searchsorted(self.time_grid.times, query, side="right") - 1
        indices = jnp.minimum(indices, self.time_grid.num_steps - 1)
        return jnp.take(values, indices, axis=len(cases))

    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.evaluate(coefficients, times, case_shape=case_shape)


class PiecewiseLinearControlParameterization(AbstractControlParameterization):
    """Continuous nodal controls linearly interpolated in physical time."""

    time_grid: TimeGrid

    def __init__(
        self,
        time_grid: TimeGrid,
        control_shape: Sequence[int],
        /,
        *,
        parameterization_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        shape = _shape(control_shape, "control_shape")
        self.time_grid = time_grid
        self.control_shape = shape
        self.parameter_shape = (time_grid.num_times,) + shape
        self.parameterization_id = _identifier(parameterization_id, "parameterization_id")
        self.approximation_id = "control:piecewise-linear:nodal"

    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        del state
        cases = _case_shape(case_shape)
        values = _coefficient_array(coefficients, cases, self.parameter_shape)
        query = _query(time)
        query = eqx.error_if(
            query,
            jnp.any(~jnp.isfinite(query))
            | jnp.any(query < self.time_grid.t0)
            | jnp.any(query > self.time_grid.t1),
            "Piecewise-linear control time lies outside its physical grid.",
        )
        lower_indices = jnp.searchsorted(self.time_grid.times, query, side="right") - 1
        lower_indices = jnp.minimum(lower_indices, self.time_grid.num_steps - 1)
        upper_indices = lower_indices + 1
        lower_time = self.time_grid.times[lower_indices]
        upper_time = self.time_grid.times[upper_indices]
        fraction = (query - lower_time) / (upper_time - lower_time)
        lower = jnp.take(values, lower_indices, axis=len(cases))
        upper = jnp.take(values, upper_indices, axis=len(cases))
        payload_ndim = len(self.control_shape)
        weight = fraction.reshape(
            (1,) * len(cases) + fraction.shape + (1,) * payload_ndim
        )
        return lower + weight * (upper - lower)

    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.evaluate(coefficients, times, case_shape=case_shape)


class BSplineControlBoundCertificate(StrictModule):
    """Convex-hull certificate for fixed-grid B-spline control bounds."""

    lower_bound: Array
    upper_bound: Array
    coefficient_minimum: Array
    coefficient_maximum: Array
    certified: Array
    parameterization_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    continuous_domain: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        lower_bound: ArrayLike,
        upper_bound: ArrayLike,
        coefficient_minimum: ArrayLike,
        coefficient_maximum: ArrayLike,
        certified: ArrayLike,
        parameterization_id: str,
    ):
        self.lower_bound = jnp.asarray(lower_bound)
        self.upper_bound = jnp.asarray(upper_bound)
        self.coefficient_minimum = jnp.asarray(coefficient_minimum)
        self.coefficient_maximum = jnp.asarray(coefficient_maximum)
        self.certified = jnp.asarray(certified, dtype=jnp.bool_)
        self.parameterization_id = _identifier(
            parameterization_id, "certificate parameterization_id"
        )
        self.certificate_id = "control-bound:bspline-convex-hull"
        self.continuous_domain = True


class BSplineControlRefinement(StrictModule):
    """A diagnosed B-spline grid transfer and its refined coefficients."""

    parameterization: BSplineControlParameterization
    coefficients: Array = fixed_field()
    transfer: BSplineGridTransfer
    source_parameterization_id: str = eqx.field(static=True)
    target_parameterization_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        parameterization: BSplineControlParameterization,
        coefficients: ArrayLike,
        transfer: BSplineGridTransfer,
        source_parameterization_id: str,
    ):
        self.parameterization = parameterization
        self.coefficients = jnp.asarray(coefficients)
        self.transfer = transfer
        self.source_parameterization_id = _identifier(
            source_parameterization_id, "source_parameterization_id"
        )
        self.target_parameterization_id = parameterization.parameterization_id


class BSplineControlParameterization(AbstractControlParameterization):
    """Differentiable fixed-grid B-spline control in physical time."""

    grid: BSplineGrid
    time_grid: TimeGrid | None

    def __init__(
        self,
        grid: BSplineGrid,
        control_shape: Sequence[int],
        /,
        *,
        parameterization_id: str,
    ):
        if not isinstance(grid, BSplineGrid):
            raise TypeError("grid must be a BSplineGrid.")
        shape = _shape(control_shape, "control_shape")
        self.grid = grid
        self.time_grid = None
        self.control_shape = shape
        self.parameter_shape = (grid.coefficient_count,) + shape
        self.parameterization_id = _identifier(parameterization_id, "parameterization_id")
        self.approximation_id = f"control:bspline:fixed-grid:degree-{grid.degree}"

    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        del state
        cases = _case_shape(case_shape)
        values = _coefficient_array(coefficients, cases, self.parameter_shape)
        query = _query(time)
        case_query = jnp.broadcast_to(query, cases + query.shape)
        return bspline_evaluate(
            self.grid.knots,
            values,
            case_query,
            degree=self.grid.degree,
            bounds="error",
            case_shape=cases,
        ).values

    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.evaluate(coefficients, times, case_shape=case_shape)

    def bound_certificate(
        self,
        coefficients: ArrayLike,
        lower_bound: ArrayLike,
        upper_bound: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> BSplineControlBoundCertificate:
        """Certify continuous bounds from the nonnegative partition of unity."""
        cases = _case_shape(case_shape)
        values = _coefficient_array(coefficients, cases, self.parameter_shape)
        lower = jnp.asarray(lower_bound)
        upper = jnp.asarray(upper_bound)
        if lower.shape not in ((), self.control_shape):
            raise ValueError("lower_bound must be scalar or have control_shape.")
        if upper.shape not in ((), self.control_shape):
            raise ValueError("upper_bound must be scalar or have control_shape.")
        lower = jnp.broadcast_to(lower, self.control_shape)
        upper = jnp.broadcast_to(upper, self.control_shape)
        lower = eqx.error_if(
            lower,
            jnp.any(~jnp.isfinite(lower))
            | jnp.any(~jnp.isfinite(upper))
            | jnp.any(lower > upper),
            "B-spline control bounds must be finite and ordered.",
        )
        coefficient_axis = len(cases)
        minimum = jnp.min(values, axis=coefficient_axis)
        maximum = jnp.max(values, axis=coefficient_axis)
        control_axes = tuple(range(len(cases), minimum.ndim))
        within = (minimum >= lower) & (maximum <= upper)
        certified = jnp.all(within, axis=control_axes) if control_axes else within
        return BSplineControlBoundCertificate(
            lower_bound=lower,
            upper_bound=upper,
            coefficient_minimum=minimum,
            coefficient_maximum=maximum,
            certified=certified,
            parameterization_id=self.parameterization_id,
        )

    def refine(
        self,
        new_grid: BSplineGrid,
        coefficients: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        parameterization_id: str,
        method: ProjectionMethod = "auto",
        maximum_condition: float = 1.0e12,
    ) -> BSplineControlRefinement:
        """Transfer coefficients through the canonical diagnosed grid transfer."""
        if not isinstance(new_grid, BSplineGrid):
            raise TypeError("new_grid must be a BSplineGrid.")
        cases = _case_shape(case_shape)
        values = _coefficient_array(coefficients, cases, self.parameter_shape)
        transfer = BSplineGridTransfer(
            self.grid,
            new_grid,
            method=method,
            maximum_condition=maximum_condition,
        )
        refined = transfer(values, coefficient_axis=len(cases))
        parameterization = BSplineControlParameterization(
            new_grid,
            self.control_shape,
            parameterization_id=parameterization_id,
        )
        return BSplineControlRefinement(
            parameterization=parameterization,
            coefficients=refined,
            transfer=transfer,
            source_parameterization_id=self.parameterization_id,
        )


def _model_value_shape(size: Any, /) -> tuple[int, ...]:
    if size == "scalar":
        return ()
    if isinstance(size, int):
        return (int(size),)
    return tuple(size)


class NeuralFeedbackPolicy(AbstractControlParameterization):
    """Learned stationary state feedback ``u(t, x) = model(x)``.

    The model reads one flat point per case: the current state flattened,
    followed by the physical time when `time_input` is set. Its sizes are
    exact: `in_size` is `prod(state_shape)` (plus one with `time_input`) and
    `out_size` produces `control_shape`; it must use a pointwise flat binding and
    is evaluated without a key. The current state is required, times are
    scalar, and the coefficient shape is empty: coefficients are a
    `case_shape` token because the decision lives in the model's arrays.
    Feedback policies cannot be sampled without a state trajectory.

    The model is a dynamic child whose arrays keep their own roles, so a
    learned policy stays PARAMETER. It is bound to the
    `AbstractControlParameterization` `DECISION` slot; `component_contract()`
    returns the bound contract.
    """

    model: AbstractArrayModel
    time_grid: TimeGrid | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    time_input: bool = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        /,
        *,
        state_shape: Sequence[int],
        control_shape: Sequence[int],
        policy_id: str,
        time_input: bool = False,
    ):
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must be an AbstractArrayModel.")
        if not isinstance(time_input, bool):
            raise TypeError("time_input must be bool.")
        states = _shape(state_shape, "state_shape")
        controls = _shape(control_shape, "control_shape")
        binding = model.input_binding()
        if binding.batch_mode != "pointwise" or binding.input_mode != "flat":
            raise ValueError(
                "Neural feedback policies require a pointwise flat model binding."
            )
        input_size = prod(states) + int(time_input)
        if model.in_size != input_size:
            raise ValueError(
                f"Neural feedback policy in_size must be {input_size}; got "
                f"{model.in_size!r}."
            )
        if _model_value_shape(model.out_size) != controls:
            raise ValueError(
                f"Neural feedback policy out_size must produce shape {controls}; got "
                f"{model.out_size!r}."
            )
        bind_component(model, AbstractControlParameterization)
        self.model = model
        self.time_grid = None
        self.state_shape = states
        self.time_input = time_input
        self.control_shape = controls
        self.parameter_shape = ()
        self.parameterization_id = _identifier(policy_id, "policy_id")
        self.approximation_id = (
            "control:neural-state-feedback:time-dependent"
            if time_input
            else "control:neural-state-feedback:stationary"
        )

    def component_contract(self) -> ComponentContract:
        """Return the model's contract bound to the control-parameterization slot."""
        return bind_component(self.model, AbstractControlParameterization).contract()

    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        cases = _case_shape(case_shape)
        _coefficient_array(coefficients, cases, self.parameter_shape)
        query = _query(time)
        if query.shape != ():
            raise ValueError("NeuralFeedbackPolicy.evaluate requires a scalar time.")
        if state is None:
            raise ValueError("NeuralFeedbackPolicy.evaluate requires the current state.")
        state_ = jnp.asarray(state)
        expected_state = cases + self.state_shape
        if tuple(state_.shape) != expected_state:
            raise ValueError(
                f"Feedback state must have shape {expected_state}; got {state_.shape}."
            )
        if not jnp.issubdtype(state_.dtype, jnp.inexact):
            state_ = state_.astype("float64")
        state_ = eqx.error_if(
            state_, jnp.any(~jnp.isfinite(state_)), "Feedback state must be finite."
        )
        points = state_.reshape((-1, prod(self.state_shape)))
        if self.time_input:
            times = jnp.broadcast_to(query.astype(points.dtype), (points.shape[0], 1))
            points = jnp.concatenate((points, times), axis=-1)
        binding = self.model.input_binding()
        values = jax.vmap(
            lambda point: binding.call(self.model, point, key=None, iter_=None, kwargs={})
        )(points)
        return values.reshape(cases + self.control_shape)

    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        del coefficients, times, case_shape
        raise ValueError(
            "NeuralFeedbackPolicy cannot be sampled without states; evaluate it "
            "online or roll it out through ControlProblem."
        )


__all__ = [
    "AbstractControlParameterization",
    "BSplineControlBoundCertificate",
    "BSplineControlParameterization",
    "BSplineControlRefinement",
    "NeuralFeedbackPolicy",
    "PiecewiseConstantControlParameterization",
    "PiecewiseLinearControlParameterization",
]

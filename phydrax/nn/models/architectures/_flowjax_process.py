#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.distributions import (
    AbstractDistribution as FlowJAXDistribution,
    Normal as FlowJAXNormal,
)
from flowjax.flows import coupling_flow
from jaxtyping import Array, ArrayLike, Key

from ...._strict import StrictModule
from ....stochastic._process import (
    AbstractMarginalTransitionLaw,
    AbstractProcessDistribution,
)


def _shape(values: Sequence[int], /, *, name: str) -> tuple[int, ...]:
    result = tuple(int(size) for size in values)
    if not result or any(size <= 0 for size in result):
        raise ValueError(f"{name} must contain positive dimensions.")
    return result


def _sample_shape(values: Sequence[int], /) -> tuple[int, ...]:
    result = tuple(int(size) for size in values)
    if any(size <= 0 for size in result):
        raise ValueError("FlowJAX process sample dimensions must be positive.")
    return result


def _check_state_shape(array: Array, state_shape: tuple[int, ...], /) -> None:
    if (
        array.ndim < len(state_shape)
        or tuple(array.shape[-len(state_shape) :]) != state_shape
    ):
        raise ValueError(
            f"FlowJAX coefficient states must end in shape {state_shape}; "
            f"got {array.shape}."
        )


def _flow_process_fingerprint(
    flow: FlowJAXDistribution,
    /,
    *,
    state_shape: tuple[int, ...],
    label: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"phydrax-flowjax-coefficient-process\0")
    digest.update(repr(state_shape).encode("ascii"))
    digest.update(repr(label).encode("utf-8"))
    for leaf in jax.tree_util.tree_leaves(flow):
        if eqx.is_array(leaf):
            value = jax.device_get(leaf)
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        else:
            digest.update(repr(leaf).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class StateTimeProcessConditioner(StrictModule):
    """Flatten the current latent state and append the two transition times."""

    state_shape: tuple[int, ...] = eqx.field(static=True)
    condition_size: int = eqx.field(static=True)

    def __init__(self, state_shape: Sequence[int], /):
        self.state_shape = _shape(state_shape, name="state_shape")
        self.condition_size = prod(self.state_shape) + 2

    def __call__(
        self,
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(state)
        _check_state_shape(values, self.state_shape)
        batch = tuple(values.shape[: -len(self.state_shape)])
        flat = values.reshape(batch + (prod(self.state_shape),))
        start = jnp.broadcast_to(jnp.asarray(t0, dtype=values.dtype), batch + (1,))
        end = jnp.broadcast_to(jnp.asarray(t1, dtype=values.dtype), batch + (1,))
        return jnp.concatenate((flat, start, end), axis=-1)


class IdentityCoefficientTransition(StrictModule):
    """Use the current coefficient state as a residual-flow location."""

    state_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, state_shape: Sequence[int], /):
        self.state_shape = _shape(state_shape, name="state_shape")

    def __call__(
        self,
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        /,
    ) -> Array:
        del t0, t1
        values = jnp.asarray(state)
        _check_state_shape(values, self.state_shape)
        return values


class FlowJAXProcessDistribution(AbstractProcessDistribution):
    """Conditional FlowJAX marginal over one latent coefficient state."""

    center: Array
    flow: FlowJAXDistribution
    condition: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    uncertainty_source: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        center: ArrayLike,
        flow: FlowJAXDistribution,
        condition: ArrayLike,
        event_shape: Sequence[int],
        process_id: str,
    ):
        if not isinstance(flow, FlowJAXDistribution):
            raise TypeError("flow must be a FlowJAX AbstractDistribution.")
        events = _shape(event_shape, name="event_shape")
        event_size = prod(events)
        if tuple(flow.shape) != (event_size,):
            raise ValueError(
                f"FlowJAX process flow.shape must be {(event_size,)}; got {flow.shape}."
            )
        if flow.cond_shape is None:
            raise ValueError("FlowJAX process marginals must be conditional.")
        center_array = jnp.asarray(center)
        _check_state_shape(center_array, events)
        batches = tuple(center_array.shape[: -len(events)])
        condition_array = jnp.asarray(condition, dtype=center_array.dtype)
        expected_condition = batches + tuple(flow.cond_shape)
        if condition_array.shape != expected_condition:
            raise ValueError(
                f"FlowJAX process condition must have shape {expected_condition}; "
                f"got {condition_array.shape}."
            )
        if not isinstance(process_id, str) or not process_id:
            raise ValueError("process_id must be a non-empty string.")
        self.center = center_array
        self.flow = flow
        self.condition = condition_array
        self.event_shape = events
        self.batch_shape = batches
        self.uncertainty_source = "process"
        self.process_id = process_id

    @property
    def location(self) -> Array:
        return self.center

    def sample(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        shape = _sample_shape(sample_shape)
        residual = self.flow.sample(
            key,
            sample_shape=shape,
            condition=self.condition,
        )
        center = self.center.reshape(self.batch_shape + (prod(self.event_shape),))
        return (residual + center).reshape(shape + self.batch_shape + self.event_shape)

    def log_prob(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value, dtype=self.center.dtype)
        if values.shape != self.center.shape:
            raise ValueError(
                f"FlowJAX process value must have shape {self.center.shape}; "
                f"got {values.shape}."
            )
        residual = values.reshape(self.batch_shape + (prod(self.event_shape),))
        residual = residual - self.center.reshape(
            self.batch_shape + (prod(self.event_shape),)
        )
        return self.flow.log_prob(residual, condition=self.condition)


class LatentFlowJAXCoefficientProcess(AbstractMarginalTransitionLaw):
    """A learned conditional marginal law in finite coefficient space.

    This class intentionally exposes no pathwise ``realize`` method: repeated marginal
    draws do not identify one common stochastic driver and therefore cannot satisfy a
    cocycle contract by construction.
    """

    flow: FlowJAXDistribution
    conditioner: Callable[[ArrayLike, ArrayLike, ArrayLike], Array]
    location_transition: Callable[[ArrayLike, ArrayLike, ArrayLike], Array]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        flow: FlowJAXDistribution,
        conditioner: Callable[[ArrayLike, ArrayLike, ArrayLike], Array],
        location_transition: Callable[[ArrayLike, ArrayLike, ArrayLike], Array],
        /,
        *,
        state_shape: Sequence[int],
        process_id: str | None = None,
        label: str | None = None,
    ):
        if not isinstance(flow, FlowJAXDistribution):
            raise TypeError("flow must be a FlowJAX AbstractDistribution.")
        if not callable(conditioner) or not callable(location_transition):
            raise TypeError("conditioner and location_transition must be callable.")
        states = _shape(state_shape, name="state_shape")
        if tuple(flow.shape) != (prod(states),):
            raise ValueError(
                "FlowJAX event size must equal the flattened coefficient state size."
            )
        if flow.cond_shape is None:
            raise ValueError("A FlowJAX coefficient process requires a conditional flow.")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError("label must be non-empty or None.")
        resolved_id = (
            _flow_process_fingerprint(flow, state_shape=states, label=label)
            if process_id is None
            else process_id
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("process_id must be a non-empty string.")
        self.flow = flow
        self.conditioner = conditioner
        self.location_transition = location_transition
        self.state_shape = states
        self.process_id = resolved_id
        self.label = label

    def marginal_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
    ) -> FlowJAXProcessDistribution:
        values = jnp.asarray(state)
        _check_state_shape(values, self.state_shape)
        start = jnp.asarray(t0, dtype=values.dtype)
        end = jnp.asarray(t1, dtype=values.dtype)
        if start.shape != () or end.shape != ():
            raise ValueError("FlowJAX process transition times must be scalar.")
        duration = end - start
        duration = eqx.error_if(
            duration,
            ~jnp.isfinite(duration) | (duration <= 0.0),
            "FlowJAX process transitions require finite t1 > t0.",
        )
        del duration
        center = jnp.asarray(self.location_transition(values, start, end))
        condition = jnp.asarray(self.conditioner(values, start, end))
        if center.shape != values.shape:
            raise ValueError(
                "FlowJAX location_transition must preserve the coefficient state shape."
            )
        return FlowJAXProcessDistribution(
            center=center,
            flow=self.flow,
            condition=condition,
            event_shape=self.state_shape,
            process_id=self.process_id,
        )


def conditional_coupling_flow_process(
    key: Key[Array, ""],
    /,
    *,
    state_shape: Sequence[int],
    process_id: str | None = None,
    label: str | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    invert: bool = True,
) -> LatentFlowJAXCoefficientProcess:
    """Build a state-and-time-conditioned residual coupling flow process."""
    states = _shape(state_shape, name="state_shape")
    layers = int(flow_layers)
    width = int(nn_width)
    depth = int(nn_depth)
    if layers <= 0 or width <= 0 or depth <= 0:
        raise ValueError("Flow layers, width, and depth must be positive.")
    event_size = prod(states)
    if event_size < 2:
        raise ValueError("Coupling-flow coefficient processes need at least two states.")
    conditioner = StateTimeProcessConditioner(states)
    location = IdentityCoefficientTransition(states)
    base = FlowJAXNormal(
        loc=jnp.zeros((event_size,), dtype=float),
        scale=jnp.ones((event_size,), dtype=float),
    )
    flow = coupling_flow(
        key,
        base_dist=base,
        cond_dim=conditioner.condition_size,
        flow_layers=layers,
        nn_width=width,
        nn_depth=depth,
        invert=bool(invert),
    )
    return LatentFlowJAXCoefficientProcess(
        flow,
        conditioner,
        location,
        state_shape=states,
        process_id=process_id,
        label=label,
    )


__all__ = [
    "FlowJAXProcessDistribution",
    "IdentityCoefficientTransition",
    "LatentFlowJAXCoefficientProcess",
    "StateTimeProcessConditioner",
    "conditional_coupling_flow_process",
]

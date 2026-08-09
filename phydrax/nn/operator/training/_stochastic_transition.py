#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ...._frozendict import frozendict
from ...._strict import StrictModule
from ....graph import broadcast_operator_topology
from ....stochastic._jump import (
    AbstractJumpProcess,
    JumpEventBatch,
    PoissonClockRealization,
)
from ....stochastic._martingale import jump_generator_observable
from ....stochastic._process import (
    AbstractMarginalTransitionLaw,
    AbstractPathwiseTransition,
    AbstractProcessDistribution,
)
from ....stochastic._realization import (
    CompositeStochasticRealization,
    StochasticRealization,
)
from ....stochastic._trajectory import _TrajectoryRecord, StochasticTrajectory
from ....stochastic._wiener import WienerRealization
from ..data import FunctionSamples, OperatorBatch, OperatorOutputSpec
from ..distribution import (
    AbstractOperatorDistribution,
    AbstractProbabilisticOperatorModel,
)


OperatorDriverKind: TypeAlias = Literal["wiener", "jump"]
OperatorDriverQuantity: TypeAlias = Literal[
    "increment",
    "event_times",
    "event_offsets",
    "event_channels",
    "event_marks",
    "event_mask",
    "channel_counts",
]
OperatorTransitionReduction: TypeAlias = Literal["none", "mean", "sum"]
OperatorTransitionKind: TypeAlias = Literal["marginal", "pathwise", "process"]


def _name(value: str | None, /, *, owner: str, optional: bool = False) -> str | None:
    if value is None:
        if optional:
            return None
        raise ValueError(f"{owner} must be a non-empty string.")
    resolved = str(value)
    if not resolved:
        raise ValueError(f"{owner} must be a non-empty string.")
    return resolved


def _reduce(value: Array, reduction: OperatorTransitionReduction, /) -> Array:
    if reduction == "none":
        return value
    if reduction == "mean":
        return jnp.mean(value)
    if reduction == "sum":
        return jnp.sum(value)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def _replace_input_values(
    batch: OperatorBatch,
    name: str,
    values: Array,
    /,
) -> OperatorBatch:
    return eqx.tree_at(lambda item: item.inputs[name].values, batch, values)


def _case_value(value: ArrayLike, batch: OperatorBatch, /, *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape not in ((), batch.case_shape):
        raise ValueError(
            f"{name} must be scalar or have OperatorBatch case shape "
            f"{batch.case_shape}; got {array.shape}."
        )
    return array


def _broadcast_case_value(value: Array, target: Array, batch: OperatorBatch, /) -> Array:
    trailing_rank = target.ndim - len(batch.case_shape)
    if value.shape == ():
        shaped = value
    else:
        shaped = value.reshape(batch.case_shape + (1,) * trailing_rank)
    return jnp.broadcast_to(shaped, target.shape)


def _expand_batch(
    batch: OperatorBatch,
    shape: tuple[int, ...],
    axes: tuple[str, ...],
    /,
) -> OperatorBatch:
    if len(shape) != len(axes):
        raise ValueError("Process expansion shape and axes must have equal rank.")
    if any(int(size) <= 0 for size in shape):
        raise ValueError("Process expansion dimensions must be positive.")
    if len(set(axes)) != len(axes) or set(axes) & set(batch.case_axes):
        raise ValueError("Process expansion axes must be new and unique.")

    def expand_samples(samples: FunctionSamples) -> FunctionSamples:
        values = (
            None
            if samples.values is None
            else jnp.broadcast_to(
                samples.values,
                shape + tuple(samples.values.shape),
            )
        )

        def expand_geometry(
            array: Array | None,
            geometry_rank: int,
        ) -> Array | None:
            if array is None or array.ndim == geometry_rank:
                return array
            return jnp.broadcast_to(array, shape + tuple(array.shape))

        topology = samples.topology
        if topology is not None and topology.case_shape:
            topology = broadcast_operator_topology(
                topology,
                shape + batch.case_shape,
            )
        return FunctionSamples(
            values=values,
            axes=samples.axes,
            coordinates=expand_geometry(samples.coordinates, 2),
            quadrature_weights=expand_geometry(
                samples.quadrature_weights,
                len(samples.sample_shape),
            ),
            mask=expand_geometry(samples.mask, len(samples.sample_shape)),
            topology=topology,
        )

    return OperatorBatch(
        inputs={name: expand_samples(samples) for name, samples in batch.inputs.items()},
        queries={
            name: expand_samples(samples) for name, samples in batch.queries.items()
        },
        case_axes=axes + batch.case_axes,
        case_shape=shape + batch.case_shape,
    )


def _default_process_axes(rank: int, existing: Sequence[str], /) -> tuple[str, ...]:
    used = set(existing)
    axes: list[str] = []
    for index in range(rank):
        candidate = f"__phydra_process_{index}"
        if candidate in used:
            raise ValueError(f"Reserved process axis {candidate!r} is already in use.")
        axes.append(candidate)
    return tuple(axes)


def _key_fingerprint(
    key: Key[Array, ""],
    process_id: str,
    times: Array,
    initial_state: Array,
    /,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"phydrax-operator-markov-chain\0")
    digest.update(process_id.encode("utf-8"))
    key_data = np.asarray(jax.device_get(jr.key_data(key)))
    time_data = np.asarray(jax.device_get(times))
    state_data = np.asarray(jax.device_get(initial_state))
    digest.update(key_data.tobytes())
    for value in (time_data, state_data):
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


class OperatorDriverBinding(StrictModule):
    """One typed stochastic-driver field in an ``OperatorBatch``."""

    input_name: str = eqx.field(static=True)
    component: str = eqx.field(static=True)
    kind: OperatorDriverKind = eqx.field(static=True)
    quantity: OperatorDriverQuantity = eqx.field(static=True)

    def __init__(
        self,
        input_name: str,
        component: str,
        /,
        *,
        kind: OperatorDriverKind,
        quantity: OperatorDriverQuantity,
    ):
        resolved_input = _name(input_name, owner="input_name")
        resolved_component = _name(component, owner="component")
        if kind not in ("wiener", "jump"):
            raise ValueError("kind must be 'wiener' or 'jump'.")
        quantities = (
            "increment",
            "event_times",
            "event_offsets",
            "event_channels",
            "event_marks",
            "event_mask",
            "channel_counts",
        )
        if quantity not in quantities:
            raise ValueError(f"quantity must be one of {quantities}; got {quantity!r}.")
        if kind == "wiener" and quantity != "increment":
            raise ValueError("Wiener bindings require quantity='increment'.")
        if kind == "jump" and quantity == "increment":
            raise ValueError("Jump bindings cannot use quantity='increment'.")
        self.input_name = str(resolved_input)
        self.component = str(resolved_component)
        self.kind = kind
        self.quantity = quantity


class OperatorTransitionSpec(StrictModule):
    """Validated binding between transition semantics and an ``OperatorBatch``.

    The specification owns no model parameters and introduces no stochasticity. It
    identifies the evolving state, duration, optional source time, typed stochastic
    driver fields, and output query contract. Static forcing and parameter inputs are
    preserved unchanged whenever a state is advanced.
    """

    output_spec: OperatorOutputSpec
    state_input: str = eqx.field(static=True)
    duration_input: str = eqx.field(static=True)
    source_time_input: str | None = eqx.field(static=True)
    driver_bindings: tuple[OperatorDriverBinding, ...]
    query_name: str = eqx.field(static=True)
    output_field: str = eqx.field(static=True)

    def __init__(
        self,
        output_spec: OperatorOutputSpec,
        /,
        *,
        state_input: str = "state",
        duration_input: str = "duration",
        source_time_input: str | None = None,
        driver_bindings: Sequence[OperatorDriverBinding] = (),
        query_name: str = "query",
        output_field: str = "output",
    ):
        if not isinstance(output_spec, OperatorOutputSpec):
            raise TypeError("output_spec must be an OperatorOutputSpec.")
        state = _name(state_input, owner="state_input")
        duration = _name(duration_input, owner="duration_input")
        source_time = _name(
            source_time_input,
            owner="source_time_input",
            optional=True,
        )
        bindings = tuple(driver_bindings)
        if any(not isinstance(binding, OperatorDriverBinding) for binding in bindings):
            raise TypeError("driver_bindings must contain OperatorDriverBinding objects.")
        driver_inputs = tuple(binding.input_name for binding in bindings)
        inputs = (
            tuple(value for value in (state, duration, source_time) if value is not None)
            + driver_inputs
        )
        if len(set(inputs)) != len(inputs):
            raise ValueError("Transition input roles must use distinct field names.")
        identities = tuple(
            (binding.component, binding.kind, binding.quantity) for binding in bindings
        )
        if len(set(identities)) != len(identities):
            raise ValueError("Typed driver bindings must have unique semantics.")
        self.output_spec = output_spec
        self.state_input = str(state)
        self.duration_input = str(duration)
        self.source_time_input = source_time
        self.driver_bindings = bindings
        self.query_name = str(_name(query_name, owner="query_name"))
        self.output_field = str(_name(output_field, owner="output_field"))

    def state_event_shape(self, batch: OperatorBatch, /) -> tuple[int, ...]:
        query = batch.query(self.query_name)
        return query.sample_shape + self.output_spec.channel_shape

    def driver_binding(self, input_name: str | None = None, /) -> OperatorDriverBinding:
        if input_name is None:
            if len(self.driver_bindings) != 1:
                raise ValueError(
                    "A driver input name is required unless exactly one binding exists."
                )
            return self.driver_bindings[0]
        for binding in self.driver_bindings:
            if binding.input_name == input_name:
                return binding
        raise KeyError(f"Unknown transition driver input {input_name!r}.")

    def driver_event_shape(
        self,
        batch: OperatorBatch,
        input_name: str | None = None,
        /,
    ) -> tuple[int, ...]:
        binding = self.driver_binding(input_name)
        values = batch.input(binding.input_name).values
        if values is None:
            raise ValueError("The configured driver input has no values.")
        return tuple(int(size) for size in values.shape[len(batch.case_shape) :])

    def validate_batch(
        self,
        batch: OperatorBatch,
        /,
        *,
        require_driver: bool = False,
    ) -> OperatorBatch:
        if not isinstance(batch, OperatorBatch):
            raise TypeError("Transition templates must be OperatorBatch objects.")
        for input_name in (self.state_input, self.duration_input):
            if input_name not in batch.inputs:
                raise KeyError(f"Transition batch is missing input {input_name!r}.")
            if batch.input(input_name).values is None:
                raise ValueError(f"Transition input {input_name!r} has no values.")
        if self.source_time_input is not None:
            if self.source_time_input not in batch.inputs:
                raise KeyError(
                    f"Transition batch is missing input {self.source_time_input!r}."
                )
            if batch.input(self.source_time_input).values is None:
                raise ValueError("The configured source-time input has no values.")
        if require_driver and not self.driver_bindings:
            raise ValueError("Pathwise transitions require typed driver bindings.")
        for binding in self.driver_bindings:
            if binding.input_name not in batch.inputs:
                raise KeyError(
                    f"Transition batch is missing input {binding.input_name!r}."
                )
            if batch.input(binding.input_name).values is None:
                raise ValueError(
                    f"Configured driver input {binding.input_name!r} has no values."
                )
        if self.query_name not in batch.queries:
            raise KeyError(f"Transition batch is missing query {self.query_name!r}.")

        state = batch.input(self.state_input)
        query = batch.query(self.query_name)
        expected = batch.case_shape + self.state_event_shape(batch)
        assert state.values is not None
        if tuple(int(size) for size in state.values.shape) != expected:
            raise ValueError(
                f"Transition state input must have shape {expected}; "
                f"got {state.values.shape}."
            )
        if state.sample_shape != query.sample_shape:
            raise ValueError(
                "Transition state and output query must have equal sample shapes."
            )
        if state.geometry_fingerprint() != query.geometry_fingerprint():
            raise ValueError(
                "Transition state and output query must use identical physical geometry."
            )
        return batch

    def condition(
        self,
        batch: OperatorBatch,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
    ) -> OperatorBatch:
        start = _case_value(t0, batch, name="t0")
        end = _case_value(t1, batch, name="t1")
        duration = end - start
        duration = eqx.error_if(
            duration,
            jnp.any(~jnp.isfinite(duration)) | jnp.any(duration <= 0.0),
            "Operator transitions require finite t1 > t0.",
        )
        duration_template = batch.input(self.duration_input).values
        assert duration_template is not None
        conditioned = _replace_input_values(
            batch,
            self.duration_input,
            _broadcast_case_value(duration, duration_template, batch),
        )
        if self.source_time_input is not None:
            time_template = conditioned.input(self.source_time_input).values
            assert time_template is not None
            conditioned = _replace_input_values(
                conditioned,
                self.source_time_input,
                _broadcast_case_value(start, time_template, conditioned),
            )
        return conditioned

    def advance(self, batch: OperatorBatch, state: ArrayLike, /) -> OperatorBatch:
        values = jnp.asarray(state)
        expected = batch.case_shape + self.state_event_shape(batch)
        if tuple(int(size) for size in values.shape) != expected:
            raise ValueError(
                f"Advanced transition state must have shape {expected}; "
                f"got {values.shape}."
            )
        return _replace_input_values(batch, self.state_input, values)

    def batch_for_state(
        self,
        template: OperatorBatch,
        state: ArrayLike,
        /,
        *,
        process_axes: Sequence[str] | None = None,
    ) -> OperatorBatch:
        values = jnp.asarray(state)
        event_shape = self.state_event_shape(template)
        base_tail = template.case_shape + event_shape
        if (
            values.ndim < len(base_tail)
            or tuple(values.shape[-len(base_tail) :]) != base_tail
        ):
            raise ValueError(
                "Transition state must end in template case_shape + event_shape; "
                f"expected tail {base_tail}, got {values.shape}."
            )
        prefix = tuple(int(size) for size in values.shape[: -len(base_tail)])
        if not prefix:
            return self.advance(template, values)
        axes = (
            _default_process_axes(len(prefix), template.case_axes)
            if process_axes is None
            else tuple(str(axis) for axis in process_axes)
        )
        expanded = _expand_batch(template, prefix, axes)
        return self.advance(expanded, values)

    def with_driver(
        self,
        batch: OperatorBatch,
        values: ArrayLike,
        /,
        *,
        input_name: str | None = None,
    ) -> OperatorBatch:
        binding = self.driver_binding(input_name)
        template = batch.input(binding.input_name).values
        assert template is not None
        event_shape = tuple(int(size) for size in template.shape[len(batch.case_shape) :])
        array = jnp.asarray(values, dtype=template.dtype)
        if len(event_shape) == 0:
            leading = tuple(int(size) for size in array.shape)
        elif (
            array.ndim >= len(event_shape)
            and tuple(array.shape[-len(event_shape) :]) == event_shape
        ):
            leading = tuple(int(size) for size in array.shape[: -len(event_shape)])
        else:
            raise ValueError(
                f"Driver field must end in event shape {event_shape}; got {array.shape}."
            )
        if (
            len(leading) > len(batch.case_shape)
            or batch.case_shape[: len(leading)] != leading
        ):
            raise ValueError(
                "Driver field leading shape must be a prefix of the transition "
                f"case shape {batch.case_shape}; got {leading}."
            )
        shaped = array.reshape(
            leading + (1,) * (len(batch.case_shape) - len(leading)) + event_shape
        )
        return _replace_input_values(
            batch,
            binding.input_name,
            jnp.broadcast_to(shaped, template.shape),
        )

    def with_drivers(
        self,
        batch: OperatorBatch,
        values: Mapping[str, ArrayLike],
        /,
    ) -> OperatorBatch:
        expected = {binding.input_name for binding in self.driver_bindings}
        supplied = set(values)
        if supplied != expected:
            raise ValueError(
                "Driver fields must exactly match configured inputs; "
                f"missing={sorted(expected - supplied)}, "
                f"unexpected={sorted(supplied - expected)}."
            )
        conditioned = batch
        for binding in self.driver_bindings:
            conditioned = self.with_driver(
                conditioned,
                values[binding.input_name],
                input_name=binding.input_name,
            )
        return conditioned

    def validate_output(self, values: ArrayLike, batch: OperatorBatch, /) -> Array:
        return self.output_spec.validate(
            jnp.asarray(values),
            batch,
            query_name=self.query_name,
        )

    def validate_distribution(
        self,
        distribution: AbstractOperatorDistribution,
        batch: OperatorBatch,
        /,
    ) -> AbstractOperatorDistribution:
        if not isinstance(distribution, AbstractOperatorDistribution):
            raise TypeError(
                "Transition model must return an AbstractOperatorDistribution."
            )
        if distribution.uncertainty_source != "process":
            raise ValueError(
                "Operator transition distributions require process uncertainty."
            )
        if (
            distribution.case_axes != batch.case_axes
            or distribution.case_shape != batch.case_shape
        ):
            raise ValueError("Operator transition distribution changed the case layout.")
        expected_event = self.state_event_shape(batch)
        if distribution.event_shape != expected_event:
            raise ValueError(
                f"Operator transition distribution event shape must be {expected_event}; "
                f"got {distribution.event_shape}."
            )
        if distribution.output_spec.channel_shape != self.output_spec.channel_shape:
            raise ValueError("Operator transition distribution output channels changed.")
        return distribution


def _shape_prefix(
    shape: tuple[int, ...],
    tail: tuple[int, ...],
    /,
    *,
    name: str,
) -> tuple[int, ...]:
    if tail:
        if len(shape) < len(tail) or shape[-len(tail) :] != tail:
            raise ValueError(f"{name} must end in shape {tail}; got {shape}.")
        return shape[: -len(tail)]
    return shape


def _operator_process_output(
    model: Callable,
    template: OperatorBatch,
    spec: OperatorTransitionSpec,
    state: ArrayLike,
    driver_values: Mapping[str, ArrayLike],
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
) -> Array:
    expected_names = {binding.input_name for binding in spec.driver_bindings}
    supplied_names = set(driver_values)
    if supplied_names != expected_names:
        raise ValueError(
            "Driver fields must exactly match configured inputs; "
            f"missing={sorted(expected_names - supplied_names)}, "
            f"unexpected={sorted(supplied_names - expected_names)}."
        )

    state_array = jnp.asarray(state)
    state_tail = template.case_shape + spec.state_event_shape(template)
    state_prefix = _shape_prefix(
        tuple(state_array.shape),
        state_tail,
        name="Process state",
    )
    case_shape = template.case_shape
    case_rank = len(case_shape)
    layouts: dict[str, tuple[Array, tuple[int, ...], tuple[int, ...], bool]] = {}
    process_shapes = [state_prefix]
    for binding in spec.driver_bindings:
        input_values = template.input(binding.input_name).values
        assert input_values is not None
        event_shape = spec.driver_event_shape(template, binding.input_name)
        array = jnp.asarray(
            driver_values[binding.input_name],
            dtype=input_values.dtype,
        )
        leading = _shape_prefix(
            tuple(array.shape),
            event_shape,
            name=f"Driver field {binding.input_name!r}",
        )
        case_specific = (
            case_rank > 0
            and len(leading) >= case_rank
            and leading[-case_rank:] == case_shape
        )
        driver_prefix = leading[:-case_rank] if case_specific else leading
        process_shapes.append(driver_prefix)
        layouts[binding.input_name] = (
            array,
            event_shape,
            driver_prefix,
            case_specific,
        )

    process_shape = jnp.broadcast_shapes(*process_shapes)
    expanded_state = jnp.broadcast_to(state_array, process_shape + state_tail)
    batch = spec.batch_for_state(template, expanded_state)
    batch = spec.condition(batch, t0=t0, t1=t1)
    expanded_drivers: dict[str, Array] = {}
    for binding in spec.driver_bindings:
        array, event_shape, driver_prefix, case_specific = layouts[binding.input_name]
        prefix_padding = (1,) * (len(process_shape) - len(driver_prefix))
        if case_specific:
            shaped = array.reshape(
                prefix_padding + driver_prefix + case_shape + event_shape
            )
        else:
            shaped = array.reshape(
                prefix_padding + driver_prefix + (1,) * case_rank + event_shape
            )
        expanded_drivers[binding.input_name] = jnp.broadcast_to(
            shaped,
            process_shape + case_shape + event_shape,
        )
    batch = spec.with_drivers(batch, expanded_drivers)
    return spec.validate_output(model(batch, key=None), batch)


def _jump_driver_values(
    spec: OperatorTransitionSpec,
    template: OperatorBatch,
    events: JumpEventBatch | Mapping[str, JumpEventBatch],
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
) -> dict[str, Array]:
    jump_components = {
        binding.component for binding in spec.driver_bindings if binding.kind == "jump"
    }
    if isinstance(events, JumpEventBatch):
        if len(jump_components) != 1:
            raise ValueError(
                "A single JumpEventBatch requires exactly one configured jump component."
            )
        event_batches = {next(iter(jump_components)): events}
    else:
        event_batches = dict(events)
    missing = jump_components - set(event_batches)
    if missing:
        raise KeyError(f"Missing jump event components {sorted(missing)}.")

    start = jnp.asarray(t0)
    end = jnp.asarray(t1)
    if start.shape != () or end.shape != ():
        raise ValueError("Jump event segment times must be scalar.")
    if bool(~jnp.isfinite(start) | ~jnp.isfinite(end) | (end <= start)):
        raise ValueError("Jump event segments require finite t1 > t0.")

    values: dict[str, Array] = {}
    masks: dict[str, Array] = {}
    for component in jump_components:
        batch = event_batches[component]
        if not isinstance(batch, JumpEventBatch):
            raise TypeError("Jump event mappings must contain JumpEventBatch values.")
        if not bool(jnp.all(batch.successful)):
            raise ValueError(
                f"Jump event component {component!r} contains incomplete paths."
            )
        masks[component] = batch.valid & (batch.times > start) & (batch.times <= end)

    for binding in spec.driver_bindings:
        if binding.kind != "jump":
            continue
        batch = event_batches[binding.component]
        mask = masks[binding.component]
        if binding.quantity == "event_times":
            value = jnp.where(mask, batch.times, 0.0)
        elif binding.quantity == "event_offsets":
            value = jnp.where(mask, batch.times - start, 0.0)
        elif binding.quantity == "event_channels":
            value = jnp.where(mask, batch.channels, -1)
        elif binding.quantity == "event_marks":
            shaped_mask = mask.reshape(mask.shape + (1,) * len(batch.mark_shape))
            value = jnp.where(shaped_mask, batch.marks, 0)
        elif binding.quantity == "event_mask":
            value = mask
        elif binding.quantity == "channel_counts":
            event_shape = spec.driver_event_shape(template, binding.input_name)
            if len(event_shape) != 1:
                raise ValueError(
                    "channel_counts driver fields must have event shape (num_channels,)."
                )
            channels = jnp.arange(event_shape[0], dtype=batch.channels.dtype)
            value = jnp.sum(
                mask[..., None] & (batch.channels[..., None] == channels),
                axis=-2,
            )
        else:
            raise AssertionError(f"Unhandled jump quantity {binding.quantity!r}.")
        values[binding.input_name] = value
    return values


class OperatorProcessDistribution(AbstractProcessDistribution):
    """Process-distribution view of one complete-field operator distribution."""

    operator_distribution: AbstractOperatorDistribution
    event_shape: tuple[int, ...] = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    uncertainty_source: Literal["process"] = eqx.field(static=True)

    def __init__(self, distribution: AbstractOperatorDistribution, /):
        if not isinstance(distribution, AbstractOperatorDistribution):
            raise TypeError("distribution must implement AbstractOperatorDistribution.")
        if distribution.uncertainty_source != "process":
            raise ValueError(
                "Operator process distributions require process uncertainty."
            )
        self.operator_distribution = distribution
        self.event_shape = distribution.event_shape
        self.batch_shape = distribution.case_shape
        self.uncertainty_source = "process"

    @property
    def location(self) -> Array:
        return self.operator_distribution.location

    def sample(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.operator_distribution.sample(key, sample_shape)

    def log_prob(self, value: ArrayLike, /) -> Array:
        return self.operator_distribution.log_prob(jnp.asarray(value))


class OperatorMarginalTransition(AbstractMarginalTransitionLaw):
    """Complete-field neural-operator transition after marginalizing its driver."""

    model: AbstractProbabilisticOperatorModel
    template_batch: OperatorBatch
    spec: OperatorTransitionSpec
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractProbabilisticOperatorModel,
        template_batch: OperatorBatch,
        spec: OperatorTransitionSpec,
        /,
        *,
        process_id: str,
    ):
        if not isinstance(model, AbstractProbabilisticOperatorModel):
            raise TypeError(
                "OperatorMarginalTransition requires an "
                "AbstractProbabilisticOperatorModel."
            )
        if not isinstance(spec, OperatorTransitionSpec):
            raise TypeError("spec must be an OperatorTransitionSpec.")
        template = spec.validate_batch(template_batch)
        resolved_id = _name(process_id, owner="process_id")
        self.model = model
        self.template_batch = template
        self.spec = spec
        self.state_shape = spec.state_event_shape(template)
        self.process_id = str(resolved_id)

    def marginal_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
    ) -> OperatorProcessDistribution:
        batch = self.spec.batch_for_state(self.template_batch, state)
        batch = self.spec.condition(batch, t0=t0, t1=t1)
        distribution = self.model.distribution(batch, key=None)
        validated = self.spec.validate_distribution(distribution, batch)
        return OperatorProcessDistribution(validated)


class OperatorPathwiseTransition(AbstractPathwiseTransition):
    """Neural-operator transition conditioned on one explicit additive driver segment."""

    model: Callable
    template_batch: OperatorBatch
    spec: OperatorTransitionSpec
    state_shape: tuple[int, ...] = eqx.field(static=True)
    driver_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: Callable,
        template_batch: OperatorBatch,
        spec: OperatorTransitionSpec,
        /,
        *,
        process_id: str,
    ):
        if not callable(model):
            raise TypeError("OperatorPathwiseTransition model must be callable.")
        if not isinstance(spec, OperatorTransitionSpec):
            raise TypeError("spec must be an OperatorTransitionSpec.")
        template = spec.validate_batch(template_batch, require_driver=True)
        if len(spec.driver_bindings) != 1:
            raise ValueError(
                "OperatorPathwiseTransition requires exactly one driver binding."
            )
        binding = spec.driver_bindings[0]
        if binding.kind != "wiener" or binding.quantity != "increment":
            raise ValueError(
                "OperatorPathwiseTransition requires one Wiener increment binding."
            )
        resolved_id = _name(process_id, owner="process_id")
        self.model = model
        self.template_batch = template
        self.spec = spec
        self.state_shape = spec.state_event_shape(template)
        self.driver_shape = spec.driver_event_shape(template)
        self.process_id = str(resolved_id)

    def pathwise_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        driver_increment: ArrayLike,
    ) -> Array:
        binding = self.spec.driver_bindings[0]
        return _operator_process_output(
            self.model,
            self.template_batch,
            self.spec,
            state,
            {binding.input_name: driver_increment},
            t0=t0,
            t1=t1,
        )

    def combine_driver_segments(
        self,
        first: ArrayLike,
        second: ArrayLike,
        /,
    ) -> Array:
        left = jnp.asarray(first)
        right = jnp.asarray(second, dtype=left.dtype)
        if left.shape != right.shape:
            raise ValueError("Additive driver segments must have equal shapes.")
        _shape_prefix(
            tuple(left.shape),
            self.driver_shape,
            name="Driver segment",
        )
        return left + right


class OperatorProcessTransition(StrictModule):
    """Operator transition conditioned on multiple typed process-driver fields."""

    model: Callable
    template_batch: OperatorBatch
    spec: OperatorTransitionSpec
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: Callable,
        template_batch: OperatorBatch,
        spec: OperatorTransitionSpec,
        /,
        *,
        process_id: str,
    ):
        if not callable(model):
            raise TypeError("OperatorProcessTransition model must be callable.")
        if not isinstance(spec, OperatorTransitionSpec):
            raise TypeError("spec must be an OperatorTransitionSpec.")
        template = spec.validate_batch(template_batch, require_driver=True)
        resolved_id = _name(process_id, owner="process_id")
        self.model = model
        self.template_batch = template
        self.spec = spec
        self.state_shape = spec.state_event_shape(template)
        self.process_id = str(resolved_id)

    def process_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        driver_values: Mapping[str, ArrayLike],
    ) -> Array:
        return _operator_process_output(
            self.model,
            self.template_batch,
            self.spec,
            state,
            driver_values,
            t0=t0,
            t1=t1,
        )


class OperatorJumpTransition(StrictModule):
    """Operator transition conditioned on canonical finite-activity jump events."""

    model: Callable
    template_batch: OperatorBatch
    spec: OperatorTransitionSpec
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: Callable,
        template_batch: OperatorBatch,
        spec: OperatorTransitionSpec,
        /,
        *,
        process_id: str,
    ):
        if not callable(model):
            raise TypeError("OperatorJumpTransition model must be callable.")
        if not isinstance(spec, OperatorTransitionSpec):
            raise TypeError("spec must be an OperatorTransitionSpec.")
        template = spec.validate_batch(template_batch, require_driver=True)
        if any(binding.kind != "jump" for binding in spec.driver_bindings):
            raise ValueError("OperatorJumpTransition accepts only jump bindings.")
        resolved_id = _name(process_id, owner="process_id")
        self.model = model
        self.template_batch = template
        self.spec = spec
        self.state_shape = spec.state_event_shape(template)
        self.process_id = str(resolved_id)

    def jump_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        events: JumpEventBatch | Mapping[str, JumpEventBatch],
    ) -> Array:
        values = _jump_driver_values(
            self.spec,
            self.template_batch,
            events,
            t0=t0,
            t1=t1,
        )
        return _operator_process_output(
            self.model,
            self.template_batch,
            self.spec,
            state,
            values,
            t0=t0,
            t1=t1,
        )


class StochasticOperatorRollout(StrictModule):
    """Canonical stochastic trajectory produced by one operator transition law."""

    trajectory: StochasticTrajectory
    metadata: frozendict[str, Any]
    process_id: str = eqx.field(static=True)
    kind: OperatorTransitionKind = eqx.field(static=True)

    def __init__(
        self,
        trajectory: StochasticTrajectory,
        /,
        *,
        process_id: str,
        kind: OperatorTransitionKind,
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(trajectory, StochasticTrajectory):
            raise TypeError("trajectory must be a StochasticTrajectory.")
        if kind not in ("marginal", "pathwise", "process"):
            raise ValueError("kind must be 'marginal', 'pathwise', or 'process'.")
        resolved_id = _name(process_id, owner="process_id")
        self.trajectory = trajectory
        self.process_id = str(resolved_id)
        self.kind = kind
        self.metadata = frozendict({} if metadata is None else metadata)

    @property
    def states(self) -> Array:
        return self.trajectory.states

    @property
    def times(self) -> Array:
        return self.trajectory.times

    @property
    def is_pathwise(self) -> bool:
        return self.kind != "marginal"

    def to_predictive(self):
        """Return a coordinate-aware predictive field with process sample axes."""
        return self.trajectory.to_predictive()


def _time_grid(times: ArrayLike, /) -> Array:
    values = jnp.asarray(times)
    if values.ndim != 1 or int(values.shape[0]) < 2:
        raise ValueError("Operator rollouts require at least two one-dimensional times.")
    if bool(jnp.any(~jnp.isfinite(values)) | jnp.any(jnp.diff(values) <= 0.0)):
        raise ValueError("Operator rollout times must be finite and strictly increasing.")
    return values


def _resolved_case_ids(
    batch: OperatorBatch,
    case_ids: Sequence[str] | None,
    parameter_ids: Sequence[str | None] | None,
    /,
) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    count = prod(batch.case_shape) if batch.case_shape else 1
    cases = (
        tuple(f"operator-case:{index}" for index in range(count))
        if case_ids is None
        else tuple(str(value) for value in case_ids)
    )
    if (
        len(cases) != count
        or any(not value for value in cases)
        or len(set(cases)) != count
    ):
        raise ValueError("case_ids must contain one unique non-empty ID per case.")
    parameters = (
        (None,) * count
        if parameter_ids is None
        else tuple(None if value is None else str(value) for value in parameter_ids)
    )
    if len(parameters) != count or any(value == "" for value in parameters):
        raise ValueError("parameter_ids must align with physical cases.")
    return cases, parameters


def _resolved_state_axes(
    law: (
        OperatorMarginalTransition
        | OperatorPathwiseTransition
        | OperatorProcessTransition
        | OperatorJumpTransition
    ),
    state_axes: Sequence[str] | None,
    /,
) -> tuple[str, ...]:
    if state_axes is not None:
        axes = tuple(str(value) for value in state_axes)
    else:
        query = law.template_batch.query(law.spec.query_name)
        sample_axes = query.axis_names if query.axes else ("point",)
        channel_axes = () if law.spec.output_spec.channels == "scalar" else ("component",)
        axes = sample_axes + channel_axes
    if len(axes) != len(law.state_shape) or any(not value for value in axes):
        raise ValueError(
            f"state_axes must contain {len(law.state_shape)} non-empty names."
        )
    if len(set(axes)) != len(axes):
        raise ValueError("state_axes must be unique.")
    return axes


def _trajectory_from_steps(
    law: (
        OperatorMarginalTransition
        | OperatorPathwiseTransition
        | OperatorProcessTransition
        | OperatorJumpTransition
    ),
    times: Array,
    steps: Sequence[Array],
    realization_shape: tuple[int, ...],
    realization_axes: tuple[str, ...],
    /,
    *,
    driver: StochasticRealization | None,
    case_ids: Sequence[str] | None,
    parameter_ids: Sequence[str | None] | None,
    state_axes: Sequence[str] | None,
    discretization_id: str | None,
    metadata: Mapping[str, Any],
) -> StochasticTrajectory:
    batch = law.template_batch
    case_rank = len(batch.case_shape)
    realization_rank = len(realization_shape)
    stacked = jnp.stack(tuple(steps), axis=realization_rank + case_rank)
    time_position = realization_rank + case_rank
    permutation = (
        tuple(range(realization_rank, realization_rank + case_rank))
        + tuple(range(realization_rank))
        + (time_position,)
        + tuple(range(time_position + 1, stacked.ndim))
    )
    states = jnp.transpose(stacked, permutation)
    cases, parameters = _resolved_case_ids(batch, case_ids, parameter_ids)
    case_count = prod(batch.case_shape) if batch.case_shape else 1
    query = batch.query(law.spec.query_name)
    resolved_discretization = (
        query.geometry_fingerprint()
        if discretization_id is None
        else str(discretization_id)
    )
    resolved_state_axes = _resolved_state_axes(law, state_axes)
    record = _TrajectoryRecord(
        times,
        states,
        state_shape=tuple(states.shape[-len(resolved_state_axes) :]),
        case_shape=batch.case_shape,
        realization_shape=realization_shape,
        realizations=(
            (driver,) * case_count
            if driver is not None
            else (None,) * case_count
        ),
        case_ids=cases,
        parameter_ids=parameters,
        discretization_id=resolved_discretization,
        approximation_id=law.process_id,
        metadata=metadata,
    )
    return record.to_stochastic_trajectory(
        case_axes=batch.case_axes,
        realization_axes=realization_axes,
        state_axes=resolved_state_axes,
    )


def marginal_operator_rollout(
    law: OperatorMarginalTransition,
    times: ArrayLike,
    /,
    *,
    key: Key[Array, ""],
    num_realizations: int,
    initial_state: ArrayLike | None = None,
    realization_axis: str = "__phydra_uq_process",
    state_axes: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    parameter_ids: Sequence[str | None] | None = None,
    discretization_id: str | None = None,
) -> StochasticOperatorRollout:
    """Sample a marginal Markov chain without claiming a common driving path."""
    if not isinstance(law, OperatorMarginalTransition):
        raise TypeError("law must be an OperatorMarginalTransition.")
    count = int(num_realizations)
    if count <= 0:
        raise ValueError("num_realizations must be positive.")
    axis = str(_name(realization_axis, owner="realization_axis"))
    if axis in law.template_batch.case_axes:
        raise ValueError("realization_axis must be distinct from physical case axes.")
    query_times = _time_grid(times)
    template_state = law.template_batch.input(law.spec.state_input).values
    assert template_state is not None
    initial = template_state if initial_state is None else jnp.asarray(initial_state)
    expected = law.template_batch.case_shape + law.state_shape
    if initial.shape != expected:
        raise ValueError(
            f"initial_state must have shape {expected}; got {initial.shape}."
        )
    state = jnp.broadcast_to(initial, (count,) + expected)
    states = [state]
    sample_keys = jr.split(key, int(query_times.shape[0]) - 1)
    for index, sample_key in enumerate(sample_keys):
        distribution = law.marginal_transition(
            state,
            t0=query_times[index],
            t1=query_times[index + 1],
        )
        state = distribution.sample(sample_key)
        states.append(state)
    chain_id = _key_fingerprint(key, law.process_id, query_times, initial)
    metadata = {
        "process_id": law.process_id,
        "operator_transition_kind": "marginal",
        "marginal_chain_id": chain_id,
        "uncertainty_source": "process",
        "pathwise": False,
    }
    trajectory = _trajectory_from_steps(
        law,
        query_times,
        states,
        (count,),
        (axis,),
        driver=None,
        case_ids=case_ids,
        parameter_ids=parameter_ids,
        state_axes=state_axes,
        discretization_id=discretization_id,
        metadata=metadata,
    )
    return StochasticOperatorRollout(
        trajectory,
        process_id=law.process_id,
        kind="marginal",
        metadata=metadata,
    )


def pathwise_operator_rollout(
    law: OperatorPathwiseTransition,
    driver: WienerRealization,
    times: ArrayLike,
    /,
    *,
    initial_state: ArrayLike | None = None,
    realization_axes: Sequence[str] | None = None,
    state_axes: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    parameter_ids: Sequence[str | None] | None = None,
    discretization_id: str | None = None,
) -> StochasticOperatorRollout:
    """Roll a driver-conditioned operator on one replayable global Wiener field."""
    if not isinstance(law, OperatorPathwiseTransition):
        raise TypeError("law must be an OperatorPathwiseTransition.")
    if not isinstance(driver, WienerRealization):
        raise TypeError("driver must be a WienerRealization.")
    if driver.levy_area != "brownian":
        raise ValueError("Pathwise operator rollouts require Brownian increments.")
    shared_driver_shape = law.driver_shape
    case_driver_shape = law.template_batch.case_shape + law.driver_shape
    if driver.noise_shape == shared_driver_shape:
        driver_case_mode = "shared"
    elif law.template_batch.case_shape and driver.noise_shape == case_driver_shape:
        driver_case_mode = "case_specific"
    else:
        raise ValueError(
            "Driver noise shape must be "
            f"{shared_driver_shape} for a shared field or {case_driver_shape} "
            f"for case-specific fields; got {driver.noise_shape}."
        )
    query_times = _time_grid(times)
    if (
        float(query_times[0]) < driver.support[0]
        or float(query_times[-1]) > driver.support[1]
    ):
        raise ValueError("Pathwise rollout times must lie inside driver support.")
    axes = (
        tuple(f"__phydra_uq_process_{index}" for index in range(len(driver.sample_shape)))
        if realization_axes is None
        else tuple(str(axis) for axis in realization_axes)
    )
    if len(axes) != len(driver.sample_shape):
        raise ValueError("realization_axes must match driver.sample_shape rank.")
    if set(axes) & set(law.template_batch.case_axes):
        raise ValueError("Realization and physical case axes must be distinct.")
    template_state = law.template_batch.input(law.spec.state_input).values
    assert template_state is not None
    initial = template_state if initial_state is None else jnp.asarray(initial_state)
    expected = law.template_batch.case_shape + law.state_shape
    if initial.shape != expected:
        raise ValueError(
            f"initial_state must have shape {expected}; got {initial.shape}."
        )
    state = jnp.broadcast_to(initial, driver.sample_shape + expected)
    states = [state]
    increments = driver.increments(
        query_times[:-1],
        query_times[1:],
        dtype=initial.real.dtype,
    )
    interval_axis = len(driver.sample_shape)
    for index in range(int(query_times.shape[0]) - 1):
        increment = jnp.take(increments, index, axis=interval_axis)
        state = law.pathwise_transition(
            state,
            t0=query_times[index],
            t1=query_times[index + 1],
            driver_increment=increment,
        )
        states.append(state)
    metadata = {
        "process_id": law.process_id,
        "operator_transition_kind": "pathwise",
        "wiener_realization_id": driver.realization_id,
        "coupling_id": driver.coupling_id,
        "uncertainty_source": "process",
        "driver_case_mode": driver_case_mode,
        "pathwise": True,
    }
    trajectory = _trajectory_from_steps(
        law,
        query_times,
        states,
        driver.sample_shape,
        axes,
        driver=driver,
        case_ids=case_ids,
        parameter_ids=parameter_ids,
        state_axes=state_axes,
        discretization_id=discretization_id,
        metadata=metadata,
    )
    return StochasticOperatorRollout(
        trajectory,
        process_id=law.process_id,
        kind="pathwise",
        metadata=metadata,
    )


def process_operator_rollout(
    law: OperatorProcessTransition | OperatorJumpTransition,
    realization: CompositeStochasticRealization,
    times: ArrayLike,
    /,
    *,
    jump_events: Mapping[str, JumpEventBatch] | None = None,
    initial_state: ArrayLike | None = None,
    realization_axes: Sequence[str] | None = None,
    state_axes: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    parameter_ids: Sequence[str | None] | None = None,
    discretization_id: str | None = None,
) -> StochasticOperatorRollout:
    """Roll typed Wiener/jump operator inputs from one composite realization."""
    if not isinstance(law, (OperatorProcessTransition, OperatorJumpTransition)):
        raise TypeError(
            "law must be an OperatorProcessTransition or OperatorJumpTransition."
        )
    if not isinstance(realization, CompositeStochasticRealization):
        raise TypeError("realization must be a CompositeStochasticRealization.")
    query_times = _time_grid(times)
    if (
        float(query_times[0]) < realization.support[0]
        or float(query_times[-1]) > realization.support[1]
    ):
        raise ValueError("Process rollout times must lie inside realization support.")

    axes = (
        tuple(
            f"__phydra_uq_process_{index}"
            for index in range(len(realization.sample_shape))
        )
        if realization_axes is None
        else tuple(str(axis) for axis in realization_axes)
    )
    if len(axes) != len(realization.sample_shape):
        raise ValueError("realization_axes must match realization.sample_shape rank.")
    if len(set(axes)) != len(axes) or set(axes) & set(law.template_batch.case_axes):
        raise ValueError("Realization axes must be unique and distinct from case axes.")

    event_batches = {} if jump_events is None else dict(jump_events)
    wiener_components: dict[str, WienerRealization] = {}
    jump_components: dict[str, PoissonClockRealization] = {}
    for binding in law.spec.driver_bindings:
        component = realization.component(binding.component)
        event_shape = law.spec.driver_event_shape(
            law.template_batch,
            binding.input_name,
        )
        if binding.kind == "wiener":
            if not isinstance(component, WienerRealization):
                raise TypeError(
                    f"Component {binding.component!r} must be a WienerRealization."
                )
            if component.levy_area != "brownian":
                raise ValueError("Operator Wiener bindings require Brownian increments.")
            shared_shape = event_shape
            case_shape = law.template_batch.case_shape + event_shape
            if component.noise_shape not in (shared_shape, case_shape):
                raise ValueError(
                    f"Wiener component {binding.component!r} noise_shape must be "
                    f"{shared_shape} or {case_shape}; got {component.noise_shape}."
                )
            wiener_components[binding.component] = component
            continue
        if not isinstance(component, PoissonClockRealization):
            raise TypeError(
                f"Component {binding.component!r} must be a PoissonClockRealization."
            )
        if binding.component not in event_batches:
            raise KeyError(f"Missing JumpEventBatch for component {binding.component!r}.")
        events = event_batches[binding.component]
        if not isinstance(events, JumpEventBatch):
            raise TypeError("jump_events mappings must contain JumpEventBatch values.")
        valid_batches = (
            realization.sample_shape,
            realization.sample_shape + law.template_batch.case_shape,
        )
        if events.batch_shape not in valid_batches:
            raise ValueError(
                f"Jump event component {binding.component!r} batch shape must be "
                f"{valid_batches[0]} or {valid_batches[1]}; got {events.batch_shape}."
            )
        if binding.quantity in (
            "event_times",
            "event_offsets",
            "event_channels",
            "event_mask",
        ):
            expected_event = (events.max_events,)
        elif binding.quantity == "event_marks":
            expected_event = (events.max_events,) + events.mark_shape
        elif binding.quantity == "channel_counts":
            expected_event = (component.num_channels,)
        else:
            raise AssertionError(f"Unhandled driver quantity {binding.quantity!r}.")
        if event_shape != expected_event:
            raise ValueError(
                f"Driver input {binding.input_name!r} must have event shape "
                f"{expected_event}; got {event_shape}."
            )
        jump_components[binding.component] = component

    template_state = law.template_batch.input(law.spec.state_input).values
    assert template_state is not None
    initial = template_state if initial_state is None else jnp.asarray(initial_state)
    expected_state = law.template_batch.case_shape + law.state_shape
    if initial.shape != expected_state:
        raise ValueError(
            f"initial_state must have shape {expected_state}; got {initial.shape}."
        )
    state = jnp.broadcast_to(
        initial,
        realization.sample_shape + expected_state,
    )
    states = [state]
    increments = {
        name: component.increments(
            query_times[:-1],
            query_times[1:],
            dtype=initial.real.dtype,
        )
        for name, component in wiener_components.items()
    }
    interval_axis = len(realization.sample_shape)
    for index in range(int(query_times.shape[0]) - 1):
        start = query_times[index]
        end = query_times[index + 1]
        if isinstance(law, OperatorJumpTransition):
            state = law.jump_transition(
                state,
                t0=start,
                t1=end,
                events=event_batches,
            )
        else:
            driver_values = _jump_driver_values(
                law.spec,
                law.template_batch,
                event_batches,
                t0=start,
                t1=end,
            )
            for binding in law.spec.driver_bindings:
                if binding.kind == "wiener":
                    driver_values[binding.input_name] = jnp.take(
                        increments[binding.component],
                        index,
                        axis=interval_axis,
                    )
            state = law.process_transition(
                state,
                t0=start,
                t1=end,
                driver_values=driver_values,
            )
        states.append(state)

    metadata = {
        "process_id": law.process_id,
        "operator_transition_kind": "process",
        "stochastic_realization_id": realization.realization_id,
        "coupling_id": realization.coupling_id,
        "driver_components": tuple(realization.components),
        "jump_event_components": tuple(jump_components),
        "uncertainty_source": "process",
        "pathwise": True,
    }
    trajectory = _trajectory_from_steps(
        law,
        query_times,
        states,
        realization.sample_shape,
        axes,
        driver=realization,
        case_ids=case_ids,
        parameter_ids=parameter_ids,
        state_axes=state_axes,
        discretization_id=discretization_id,
        metadata=metadata,
    )
    return StochasticOperatorRollout(
        trajectory,
        process_id=law.process_id,
        kind="process",
        metadata=metadata,
    )


def operator_markov_chain_nll(
    law: OperatorMarginalTransition,
    states: ArrayLike,
    times: ArrayLike,
    /,
    *,
    reduction: OperatorTransitionReduction = "mean",
) -> Array:
    """Teacher-forced joint Markov-chain NLL over adjacent complete fields."""
    if not isinstance(law, OperatorMarginalTransition):
        raise TypeError("law must be an OperatorMarginalTransition.")
    query_times = _time_grid(times)
    values = jnp.asarray(states)
    expected = (
        (int(query_times.shape[0]),) + law.template_batch.case_shape + law.state_shape
    )
    if values.shape != expected:
        raise ValueError(f"states must have shape {expected}; got {values.shape}.")
    terms = []
    for index in range(int(query_times.shape[0]) - 1):
        distribution = law.marginal_transition(
            values[index],
            t0=query_times[index],
            t1=query_times[index + 1],
        )
        terms.append(-distribution.log_prob(values[index + 1]))
    return _reduce(jnp.stack(tuple(terms), axis=0), reduction)


def direct_operator_horizon_nll(
    law: OperatorMarginalTransition,
    initial_state: ArrayLike,
    targets: ArrayLike,
    /,
    *,
    initial_time: ArrayLike,
    target_times: ArrayLike,
    reduction: OperatorTransitionReduction = "mean",
) -> Array:
    """Supervise every requested horizon directly from one initial complete field."""
    if not isinstance(law, OperatorMarginalTransition):
        raise TypeError("law must be an OperatorMarginalTransition.")
    times = jnp.asarray(target_times)
    if times.ndim != 1 or int(times.shape[0]) <= 0:
        raise ValueError("target_times must be a non-empty vector.")
    state = jnp.asarray(initial_state)
    base_shape = law.template_batch.case_shape + law.state_shape
    if state.shape != base_shape:
        raise ValueError(
            f"initial_state must have shape {base_shape}; got {state.shape}."
        )
    target_values = jnp.asarray(targets)
    expected = (int(times.shape[0]),) + base_shape
    if target_values.shape != expected:
        raise ValueError(
            f"targets must have shape {expected}; got {target_values.shape}."
        )
    terms = []
    for index in range(int(times.shape[0])):
        distribution = law.marginal_transition(
            state,
            t0=initial_time,
            t1=times[index],
        )
        terms.append(-distribution.log_prob(target_values[index]))
    return _reduce(jnp.stack(tuple(terms), axis=0), reduction)


def operator_jump_generator_objective(
    law: AbstractMarginalTransitionLaw,
    process: AbstractJumpProcess,
    state: ArrayLike,
    /,
    *,
    time: ArrayLike,
    step: ArrayLike,
    observable: Callable[[Array], Array],
    key: Key[Array, ""],
    continuous_generator: Callable[[Array, Array], Array] | None = None,
    num_transition_samples: int = 256,
    num_mark_samples: int = 1,
    args: Any = None,
    reduction: OperatorTransitionReduction = "mean",
) -> Array:
    """Match a learned marginal law to a direct nonlocal jump generator."""
    if not isinstance(law, AbstractMarginalTransitionLaw):
        raise TypeError("law must implement AbstractMarginalTransitionLaw.")
    if not isinstance(process, AbstractJumpProcess):
        raise TypeError("process must implement AbstractJumpProcess.")
    if not callable(observable):
        raise TypeError("observable must be callable.")
    if continuous_generator is not None and not callable(continuous_generator):
        raise TypeError("continuous_generator must be callable or None.")
    count = int(num_transition_samples)
    if count <= 0:
        raise ValueError("num_transition_samples must be positive.")
    t0 = jnp.asarray(time)
    dt = jnp.asarray(step)
    if t0.shape != () or dt.shape != ():
        raise ValueError("time and step must be scalars.")
    dt = eqx.error_if(
        dt,
        ~jnp.isfinite(dt) | (dt <= 0.0),
        "Jump generator objectives require a finite positive step.",
    )
    state_array = jnp.asarray(state)
    transition_key, mark_key = jr.split(key)
    distribution = law.marginal_transition(state_array, t0=t0, t1=t0 + dt)
    samples = distribution.sample(transition_key, (count,))
    evolved = jax.vmap(observable)(samples)
    base = jnp.asarray(observable(state_array))
    estimate = (jnp.mean(evolved, axis=0) - base) / dt
    target = jump_generator_observable(
        process,
        state_array,
        time=t0,
        observable=observable,
        key=mark_key,
        num_mark_samples=num_mark_samples,
        args=args,
    )
    if continuous_generator is not None:
        target = target + jnp.asarray(continuous_generator(state_array, t0))
    if estimate.shape != target.shape:
        raise ValueError(
            "Empirical and declared jump-generator observables must have equal shapes."
        )
    return _reduce(jnp.abs(estimate - target) ** 2, reduction)


def operator_weak_generator_objective(
    law: AbstractMarginalTransitionLaw,
    state: ArrayLike,
    /,
    *,
    time: ArrayLike,
    step: ArrayLike,
    observable: Callable[[Array], Array],
    generator_observable: Callable[[Array, Array], Array],
    key: Key[Array, ""],
    num_samples: int = 256,
    reduction: OperatorTransitionReduction = "mean",
) -> Array:
    """Match a weak infinitesimal generator without differentiating sample paths."""
    if not isinstance(law, AbstractMarginalTransitionLaw):
        raise TypeError("law must implement AbstractMarginalTransitionLaw.")
    if not callable(observable) or not callable(generator_observable):
        raise TypeError("observable and generator_observable must be callable.")
    count = int(num_samples)
    if count <= 0:
        raise ValueError("num_samples must be positive.")
    t0 = jnp.asarray(time)
    dt = jnp.asarray(step)
    if t0.shape != () or dt.shape != ():
        raise ValueError("time and step must be scalars.")
    dt = eqx.error_if(
        dt,
        ~jnp.isfinite(dt) | (dt <= 0.0),
        "Weak generator objectives require a finite positive step.",
    )
    state_array = jnp.asarray(state)
    distribution = law.marginal_transition(state_array, t0=t0, t1=t0 + dt)
    samples = distribution.sample(key, (count,))
    evolved = jax.vmap(observable)(samples)
    base = jnp.asarray(observable(state_array))
    estimate = (jnp.mean(evolved, axis=0) - base) / dt
    target = jnp.asarray(generator_observable(state_array, t0))
    if estimate.shape != target.shape:
        raise ValueError(
            "Empirical and declared generator observables must have equal shapes."
        )
    return _reduce(jnp.abs(estimate - target) ** 2, reduction)


__all__ = [
    "direct_operator_horizon_nll",
    "marginal_operator_rollout",
    "operator_jump_generator_objective",
    "operator_markov_chain_nll",
    "operator_weak_generator_objective",
    "OperatorDriverBinding",
    "OperatorDriverKind",
    "OperatorDriverQuantity",
    "OperatorJumpTransition",
    "OperatorMarginalTransition",
    "OperatorPathwiseTransition",
    "OperatorProcessDistribution",
    "OperatorProcessTransition",
    "OperatorTransitionKind",
    "OperatorTransitionReduction",
    "OperatorTransitionSpec",
    "pathwise_operator_rollout",
    "process_operator_rollout",
    "StochasticOperatorRollout",
]

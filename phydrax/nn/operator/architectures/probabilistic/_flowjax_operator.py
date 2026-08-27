#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flowjax.distributions import (
    AbstractDistribution as FlowJAXDistribution,
    Normal as FlowJAXNormal,
)
from flowjax.flows import coupling_flow
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._frozendict import frozendict
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax._uncertainty import UncertaintySource, validate_uncertainty_source
from phydrax.nn._keys import EvalKey, split_eval_key
from phydrax.nn.operator.architectures.conditioning._deeponet import (
    AbstractBranchEncoder,
)
from phydrax.nn.operator.capabilities import ConfiguredOperatorContract
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch, OperatorOutputSpec
from phydrax.nn.operator.distribution import (
    AbstractOperatorDistribution,
    AbstractProbabilisticOperatorModel,
)
from phydrax.nn.operator.engine import AbstractOperatorModel


class _FixedReferenceQuery(StrictModule, NonTrainableState):
    """Fixed query metadata excluded from optimizer parameter partitions."""

    value: FunctionSamples

    def __init__(self, value: FunctionSamples, /):
        self.value = value


class OperatorBatchConditioner(StrictModule):
    """Concatenate named branch encodings into one condition vector per case."""

    encoders: frozendict[str, AbstractBranchEncoder]
    condition_size: int

    def __init__(self, encoders: Mapping[str, AbstractBranchEncoder], /):
        resolved: dict[str, AbstractBranchEncoder] = {}
        for name, encoder in encoders.items():
            resolved_name = str(name)
            if not resolved_name:
                raise ValueError(
                    "OperatorBatchConditioner input names must be non-empty."
                )
            if not isinstance(encoder, AbstractBranchEncoder):
                raise TypeError(
                    f"Conditioner input {resolved_name!r} must use a branch encoder."
                )
            resolved[resolved_name] = encoder
        if not resolved:
            raise ValueError("OperatorBatchConditioner requires at least one encoder.")
        self.encoders = frozendict(resolved)
        self.condition_size = sum(encoder.latent_size for encoder in resolved.values())

    def __call__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(batch, OperatorBatch):
            raise TypeError("OperatorBatchConditioner requires an OperatorBatch.")
        keys = split_eval_key(key, len(self.encoders))
        encoded: list[Array] = []
        for index, (name, encoder) in enumerate(self.encoders.items()):
            if name not in batch.inputs:
                raise KeyError(
                    f"Conditioner input {name!r} is absent from the OperatorBatch."
                )
            value = jnp.asarray(
                encoder(
                    batch.input(name),
                    case_ndim=len(batch.case_axes),
                    key=keys[index],
                )
            )
            expected = batch.case_shape + (encoder.latent_size,)
            if value.shape != expected:
                raise ValueError(
                    f"Conditioner encoder {name!r} must return shape {expected}; "
                    f"got {value.shape}."
                )
            encoded.append(value)
        return jnp.concatenate(tuple(encoded), axis=-1)


def _query_mismatch(
    reference: FunctionSamples,
    query: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    if (
        reference.sample_shape != query.sample_shape
        or len(reference.axes) != len(query.axes)
        or reference.geometry_case_shape
        or query.geometry_case_shape not in ((), case_shape)
        or (reference.topology is None) != (query.topology is None)
    ):
        return jnp.asarray(True)
    mismatch = jnp.asarray(False)
    for left, right in zip(reference.axes, query.axes, strict=True):
        if (
            left.name != right.name
            or left.basis != right.basis
            or left.periodic != right.periodic
        ):
            return jnp.asarray(True)
    mismatch = mismatch | ~jnp.allclose(
        reference.coordinates_array(case_shape=case_shape),
        query.coordinates_array(case_shape=case_shape),
        rtol=1e-6,
        atol=1e-7,
    )
    mismatch = mismatch | ~jnp.allclose(
        reference.quadrature(case_shape=case_shape),
        query.quadrature(case_shape=case_shape),
        rtol=1e-6,
        atol=1e-7,
    )
    mismatch = mismatch | ~jnp.array_equal(
        reference.mask_array(case_shape=case_shape),
        query.mask_array(case_shape=case_shape),
    )
    return mismatch


class FlowJAXOperatorDistribution(AbstractOperatorDistribution):
    """Conditional FlowJAX density over active entries of one fixed output field."""

    center: Array
    flow: FlowJAXDistribution
    condition: Array
    active_indices: Array
    query: FunctionSamples
    output_spec: OperatorOutputSpec
    case_axes: tuple[str, ...]
    case_shape: tuple[int, ...]
    uncertainty_source: UncertaintySource

    def __init__(
        self,
        *,
        center: Array,
        flow: FlowJAXDistribution,
        condition: Array,
        active_indices: Array,
        query: FunctionSamples,
        output_spec: OperatorOutputSpec,
        case_axes: tuple[str, ...],
        case_shape: tuple[int, ...],
        uncertainty_source: UncertaintySource,
    ):
        if not isinstance(flow, FlowJAXDistribution):
            raise TypeError("flow must be a FlowJAX AbstractDistribution.")
        cases = tuple(int(size) for size in case_shape)
        axes = tuple(str(axis) for axis in case_axes)
        expected = cases + query.sample_shape + output_spec.channel_shape
        center_array = jnp.asarray(center)
        if center_array.shape != expected:
            raise ValueError(
                f"FlowJAX operator center must have shape {expected}; "
                f"got {center_array.shape}."
            )
        indices = jnp.asarray(active_indices, dtype=jnp.int32)
        if indices.ndim != 1 or int(indices.shape[0]) <= 0:
            raise ValueError("FlowJAX active_indices must be a non-empty vector.")
        if tuple(flow.shape) != (int(indices.shape[0]),):
            raise ValueError(
                "FlowJAX event shape must equal the active fixed-query event size."
            )
        cond_shape = tuple(flow.cond_shape) if flow.cond_shape is not None else None
        condition_array = jnp.asarray(condition)
        if cond_shape is None or len(cond_shape) != 1:
            raise ValueError(
                "FlowJAX operator distributions must be conditionally vector-valued."
            )
        if condition_array.shape != cases + cond_shape:
            raise ValueError(
                f"FlowJAX condition must have shape {cases + cond_shape}; "
                f"got {condition_array.shape}."
            )
        if len(axes) != len(cases):
            raise ValueError("FlowJAX case axes and case shape ranks differ.")
        mask = query.mask_array(case_shape=cases)
        if output_spec.channels != "scalar":
            mask = jnp.broadcast_to(mask[..., None], expected)
        self.center = jnp.where(mask, center_array, 0.0)
        self.flow = flow
        self.condition = condition_array
        self.active_indices = indices
        self.query = query
        self.output_spec = output_spec
        self.case_axes = axes
        self.case_shape = cases
        self.uncertainty_source = validate_uncertainty_source(
            uncertainty_source,
            owner="FlowJAXOperatorDistribution uncertainty_source",
        )

    @property
    def location(self) -> Array:
        return self.center

    def sample(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        shape = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("FlowJAX operator sample dimensions must be positive.")
        cases = prod(self.case_shape) if self.case_shape else 1
        center = self.center.reshape((cases, self.event_size))[:, self.active_indices]
        condition = self.condition.reshape((cases, -1))
        keys = jr.split(key, cases)

        def draw(case_key, case_condition, case_center):
            residual = self.flow.sample(
                case_key,
                sample_shape=shape,
                condition=case_condition,
            )
            return residual + case_center

        case_values = jax.vmap(draw)(keys, condition, center)
        active_values = jnp.moveaxis(case_values, 0, len(shape))
        flat = jnp.zeros(
            shape + (cases, self.event_size),
            dtype=self.center.dtype,
        )
        flat = flat.at[..., self.active_indices].set(active_values)
        return flat.reshape(shape + self.case_shape + self.event_shape)

    def log_prob(self, target: Array, /) -> Array:
        target_array = jnp.asarray(target)
        if target_array.shape != self.center.shape:
            raise ValueError(
                f"FlowJAX operator target must have shape {self.center.shape}; "
                f"got {target_array.shape}."
            )
        cases = prod(self.case_shape) if self.case_shape else 1
        center = self.center.reshape((cases, self.event_size))[:, self.active_indices]
        target_flat = target_array.reshape((cases, self.event_size))[
            :, self.active_indices
        ]
        condition = self.condition.reshape((cases, -1))
        residual = target_flat - center
        values = jax.vmap(
            lambda value, case_condition: self.flow.log_prob(
                value,
                condition=case_condition,
            )
        )(residual, condition)
        return jnp.asarray(values).reshape(self.case_shape)


def _conditional_flow_operator_contract(model):
    wrapped = model.location_model.operator_contract
    supported_queries = tuple(
        geometry
        for geometry in wrapped.capabilities.query_geometries
        if geometry in ("tensor_grid", "point_cloud")
    )
    return ConfiguredOperatorContract(
        architecture="ConditionalFlowFunctionOperator",
        configuration=wrapped.configuration
        + (
            ("wrapped_architecture", wrapped.architecture),
            ("flow_type", type(model.flow).__name__),
            ("event_size", int(model.active_indices.shape[0])),
            ("condition_inputs", tuple(model.conditioner.encoders)),
            ("condition_size", model.conditioner.condition_size),
            ("query_fingerprint", model.reference_query_fingerprint),
            ("uncertainty_source", model.uncertainty_source),
        ),
        capabilities=replace(
            wrapped.capabilities,
            query_geometries=supported_queries,
            requires_fixed_query=True,
            multiple_queries=False,
            resolution_transfer=False,
            topology="unused",
        ),
        training=wrapped.training,
    )


class ConditionalFlowFunctionOperator(AbstractProbabilisticOperatorModel):
    """Fixed-query conditional residual flow around a deterministic operator."""

    operator_architecture = "ConditionalFlowFunctionOperator"
    _operator_contract_builder = staticmethod(_conditional_flow_operator_contract)

    location_model: AbstractOperatorModel
    conditioner: OperatorBatchConditioner
    flow: FlowJAXDistribution
    _reference_query: _FixedReferenceQuery
    active_indices: Array
    reference_query_fingerprint: str
    output_spec: OperatorOutputSpec
    uncertainty_source: UncertaintySource
    in_size: Any
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        location_model: AbstractOperatorModel,
        conditioner: OperatorBatchConditioner,
        flow: FlowJAXDistribution,
        reference_query: FunctionSamples,
        /,
        *,
        uncertainty_source: UncertaintySource,
    ):
        if not isinstance(location_model, AbstractOperatorModel):
            raise TypeError("location_model must be a neural operator.")
        if not isinstance(conditioner, OperatorBatchConditioner):
            raise TypeError("conditioner must be an OperatorBatchConditioner.")
        if not isinstance(flow, FlowJAXDistribution):
            raise TypeError("flow must be a FlowJAX AbstractDistribution.")
        if not isinstance(reference_query, FunctionSamples):
            raise TypeError("reference_query must be FunctionSamples.")
        if reference_query.geometry_case_shape:
            raise ValueError("FlowJAX requires one query geometry shared by every case.")
        if reference_query.topology is not None:
            raise ValueError(
                "FlowJAX fixed-query operators do not support native topology."
            )
        output_spec = location_model.operator_output_specs["output"]
        mask = np.asarray(reference_query.mask_array(case_shape=()), dtype=bool)
        if output_spec.channels != "scalar":
            mask = np.broadcast_to(
                mask[..., None], mask.shape + output_spec.channel_shape
            )
        active = np.flatnonzero(mask.reshape(-1))
        if active.size <= 0:
            raise ValueError(
                "FlowJAX reference queries require at least one active output."
            )
        if tuple(flow.shape) != (int(active.size),):
            raise ValueError(
                f"FlowJAX flow.shape must be {(int(active.size),)}; got {flow.shape}."
            )
        if tuple(flow.cond_shape or ()) != (conditioner.condition_size,):
            raise ValueError(
                "FlowJAX flow.cond_shape must match conditioner.condition_size."
            )
        self.location_model = location_model
        self.conditioner = conditioner
        self.flow = flow
        self._reference_query = _FixedReferenceQuery(reference_query)
        self.active_indices = jnp.asarray(active, dtype=jnp.int32)
        self.reference_query_fingerprint = reference_query.geometry_fingerprint()
        self.output_spec = output_spec
        self.uncertainty_source = validate_uncertainty_source(
            uncertainty_source,
            owner="ConditionalFlowFunctionOperator uncertainty_source",
        )
        self.in_size = location_model.in_size
        self.out_size = output_spec.channels

    @property
    def reference_query(self) -> FunctionSamples:
        return self._reference_query.value

    @property
    def operator_output_specs(self) -> dict[str, OperatorOutputSpec]:
        return {"output": self.output_spec}

    def distribution(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> FlowJAXOperatorDistribution:
        if not isinstance(batch, OperatorBatch):
            raise TypeError("ConditionalFlowFunctionOperator requires an OperatorBatch.")
        location_key, condition_key = split_eval_key(key, 2)
        location = self.output_spec.validate(
            self.location_model.__call_operator_batch__(batch, key=location_key),
            batch,
            query_name=batch.single_query_name(),
        )
        location = eqx.error_if(
            location,
            _query_mismatch(
                self.reference_query,
                batch.require_single_query(),
                batch.case_shape,
            ),
            "ConditionalFlowFunctionOperator requires its fixed reference query geometry.",
        )
        condition = self.conditioner(batch, key=condition_key)
        return FlowJAXOperatorDistribution(
            center=location,
            flow=self.flow,
            condition=condition,
            active_indices=self.active_indices,
            query=batch.require_single_query(),
            output_spec=self.output_spec,
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
            uncertainty_source=self.uncertainty_source,
        )


def conditional_coupling_flow_operator(
    key: Key[Array, ""],
    /,
    *,
    location_model: AbstractOperatorModel,
    conditioner: OperatorBatchConditioner,
    reference_query: FunctionSamples,
    uncertainty_source: UncertaintySource,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    invert: bool = True,
) -> ConditionalFlowFunctionOperator:
    """Build a conditional FlowJAX coupling-flow operator on one fixed query."""
    output_spec = location_model.operator_output_specs["output"]
    mask = np.asarray(reference_query.mask_array(case_shape=()), dtype=bool)
    if output_spec.channels != "scalar":
        mask = np.broadcast_to(mask[..., None], mask.shape + output_spec.channel_shape)
    event_size = int(np.sum(mask))
    if event_size < 2:
        raise ValueError(
            "Conditional coupling flows require at least two active outputs."
        )
    if int(flow_layers) <= 0 or int(nn_width) <= 0 or int(nn_depth) <= 0:
        raise ValueError("Flow layers, width, and depth must be positive.")
    base = FlowJAXNormal(
        loc=jnp.zeros((event_size,), dtype=float),
        scale=jnp.ones((event_size,), dtype=float),
    )
    flow = coupling_flow(
        key,
        base_dist=base,
        cond_dim=conditioner.condition_size,
        flow_layers=int(flow_layers),
        nn_width=int(nn_width),
        nn_depth=int(nn_depth),
        invert=bool(invert),
    )
    return ConditionalFlowFunctionOperator(
        location_model,
        conditioner,
        flow,
        reference_query,
        uncertainty_source=uncertainty_source,
    )


__all__ = [
    "ConditionalFlowFunctionOperator",
    "FlowJAXOperatorDistribution",
    "OperatorBatchConditioner",
    "conditional_coupling_flow_operator",
]

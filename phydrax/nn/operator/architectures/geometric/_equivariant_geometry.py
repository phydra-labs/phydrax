#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.encoded import AbstractEncodedOperatorModel
from phydrax.nn.operator.layers import (
    EquivariantIntegralLayer,
    o3_gated_activation,
    O3PointwiseLinear,
)
from phydrax.nn.operator.representations import O3Representation


class EquivariantOperatorState(StrictModule):
    """Encoded O(3) field state on the source geometry."""

    values: Array
    coordinates: Array
    weights: Array
    mask: Array
    case_shape: tuple[int, ...]

    def __init__(
        self,
        *,
        values: Array,
        coordinates: Array,
        weights: Array,
        mask: Array,
        case_shape: tuple[int, ...],
    ):
        self.values = jnp.asarray(values)
        self.coordinates = jnp.asarray(coordinates)
        self.weights = jnp.asarray(weights)
        self.mask = jnp.asarray(mask, dtype=bool)
        self.case_shape = tuple(int(size) for size in case_shape)


class EquivariantGeometryOperator(AbstractEncodedOperatorModel):
    """EqGINO-style intrinsic O(3)-equivariant operator on 3-D point geometries."""

    operator_architecture = "EqGINO"

    input_representation: O3Representation
    hidden_representation: O3Representation
    output_representation: O3Representation
    input_projection: O3PointwiseLinear
    processor_layers: tuple[EquivariantIntegralLayer, ...]
    output_layer: EquivariantIntegralLayer
    source_key: str | None
    depth: int
    in_size: int
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        input_representation: O3Representation,
        output_representation: O3Representation,
        /,
        *,
        hidden_representation: O3Representation | None = None,
        radius: float,
        radial_basis_size: int = 16,
        depth: int = 3,
        source_key: str | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.input_representation = input_representation
        self.output_representation = output_representation
        self.hidden_representation = (
            input_representation
            if hidden_representation is None
            else hidden_representation
        )
        self.source_key = source_key
        self.depth = int(depth)
        if self.depth <= 0:
            raise ValueError("Equivariant operator depth must be positive.")
        self.in_size = input_representation.packed_size
        self.out_size = (
            "scalar"
            if output_representation.packed_size == 1
            and output_representation.scalars == 1
            else output_representation.packed_size
        )
        keys = jr.split(key, self.depth + 2)
        self.input_projection = O3PointwiseLinear(
            input_representation,
            self.hidden_representation,
            key=keys[0],
        )
        self.processor_layers = tuple(
            EquivariantIntegralLayer(
                self.hidden_representation,
                self.hidden_representation,
                radius=radius,
                radial_basis_size=radial_basis_size,
                key=keys[1 + index],
            )
            for index in range(self.depth)
        )
        self.output_layer = EquivariantIntegralLayer(
            self.hidden_representation,
            output_representation,
            radius=radius,
            radial_basis_size=radial_basis_size,
            key=keys[-1],
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError(
                "EquivariantGeometryOperator requires source_key for multiple inputs."
            )
        return next(iter(batch.inputs.values()))

    def encode_inputs(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> EquivariantOperatorState:
        del key
        source = self._source(batch)
        if source.values is None:
            raise ValueError("Equivariant source requires one packed value array.")
        values = source.values
        sample_shape = source.sample_shape
        count = prod(sample_shape)
        scalar_shape = batch.case_shape + sample_shape
        if self.input_representation.packed_size == 1 and values.shape == scalar_shape:
            values = values[..., None]
        expected = scalar_shape + (self.input_representation.packed_size,)
        if tuple(int(size) for size in values.shape) != expected:
            raise ValueError(
                f"Equivariant source values must have shape {expected}; got {values.shape}."
            )
        cases = prod(batch.case_shape) if batch.case_shape else 1
        values = values.reshape((cases, count, self.input_representation.packed_size))
        coordinates = source.coordinates_array(
            case_shape=batch.case_shape, flatten=True
        ).reshape((cases, count, -1))
        if int(coordinates.shape[-1]) != 3:
            raise ValueError(
                "Equivariant geometry coordinates must be three-dimensional."
            )
        weights = source.quadrature(case_shape=batch.case_shape).reshape((cases, count))
        mask = source.mask_array(case_shape=batch.case_shape).reshape((cases, count))
        hidden = o3_gated_activation(
            self.input_projection(values), self.hidden_representation
        )
        for layer in self.processor_layers:
            update = layer(
                hidden,
                coordinates,
                coordinates,
                weights,
                source_mask=mask,
                target_mask=mask,
            )
            hidden = (
                o3_gated_activation(hidden + update, self.hidden_representation)
                * mask[..., None]
            )
        return EquivariantOperatorState(
            values=hidden,
            coordinates=coordinates,
            weights=weights,
            mask=mask,
            case_shape=batch.case_shape,
        )

    def decode_query(
        self,
        state: EquivariantOperatorState,
        query: FunctionSamples,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        cases = prod(state.case_shape) if state.case_shape else 1
        query_count = prod(query.sample_shape)
        coordinates = query.coordinates_array(
            case_shape=state.case_shape, flatten=True
        ).reshape((cases, query_count, -1))
        if int(coordinates.shape[-1]) != 3:
            raise ValueError("Equivariant query coordinates must be three-dimensional.")
        query_mask = query.mask_array(case_shape=state.case_shape).reshape(
            (cases, query_count)
        )
        output = self.output_layer(
            state.values,
            state.coordinates,
            coordinates,
            state.weights,
            source_mask=state.mask,
            target_mask=query_mask,
        )
        shaped = output.reshape(
            state.case_shape
            + query.sample_shape
            + (self.output_representation.packed_size,)
        )
        return shaped[..., 0] if self.out_size == "scalar" else shaped

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("EquivariantGeometryOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


def EqGINO(
    input_representation: O3Representation,
    output_representation: O3Representation,
    /,
    *,
    hidden_representation: O3Representation | None = None,
    radius: float,
    radial_basis_size: int = 16,
    depth: int = 3,
    source_key: str | None = None,
    key: Key[Array, ""] = DOC_KEY0,
) -> EquivariantGeometryOperator:
    """Construct the EqGINO configuration of the equivariant geometry operator."""
    return EquivariantGeometryOperator(
        input_representation,
        output_representation,
        hidden_representation=hidden_representation,
        radius=radius,
        radial_basis_size=radial_basis_size,
        depth=depth,
        source_key=source_key,
        key=key,
    )


__all__ = [
    "EqGINO",
    "EquivariantGeometryOperator",
    "EquivariantOperatorState",
]

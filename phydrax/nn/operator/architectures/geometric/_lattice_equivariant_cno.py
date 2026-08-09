#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey
from phydrax.nn.operator.data import OperatorAxis, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel
from phydrax.nn.operator.layers import (
    InvariantFilterBasis,
    LatticeEquivariantConvND,
    TensorNormActivation,
    TensorRMSNorm,
)
from phydrax.nn.operator.representations import (
    FiniteOrthogonalGroup,
    TensorFieldBlock,
    TensorFieldLayout,
    TensorType,
)


LatticeActivation = Literal["gelu", "silu", "tanh"]


def _activation(name: LatticeActivation, /):
    if name == "gelu":
        return jax.nn.gelu
    if name == "silu":
        return jax.nn.silu
    if name == "tanh":
        return jnp.tanh
    raise ValueError("activation must be 'gelu', 'silu', or 'tanh'.")


def _operator_source(batch: OperatorBatch, source_key: str | None, /):
    if source_key is not None:
        return batch.input(source_key)
    if len(batch.inputs) != 1:
        raise ValueError("source_key is required for multiple operator inputs.")
    return next(iter(batch.inputs.values()))


def _prepare_grid_values(
    values: Array,
    axes: tuple[OperatorAxis, ...],
    channels: int,
    /,
) -> Array:
    array = jnp.asarray(values)
    sample_shape = tuple(axis.size for axis in axes)
    dimension = len(sample_shape)
    if array.ndim >= dimension and tuple(array.shape[-dimension:]) == sample_shape:
        if channels != 1:
            raise ValueError(f"Expected {channels} tensor channels, got scalar values.")
        array = array[..., None]
    elif (
        array.ndim <= dimension
        or tuple(array.shape[-dimension - 1 : -1]) != sample_shape
        or int(array.shape[-1]) != channels
    ):
        raise ValueError(
            "Grid values must have case axes followed by the declared spatial and "
            f"tensor-channel shapes; got {array.shape}."
        )
    return array


def _contract_configuration(
    model: LatticeEquivariantCNO,
) -> tuple[tuple[str, object], ...]:
    return (
        ("symmetry_group", model.group.name),
        ("group_fingerprint", model.group.fingerprint),
        ("input_layout", model.input_layout.to_dict()),
        ("hidden_layout", model.hidden_layout.to_dict()),
        ("output_layout", model.output_layout.to_dict()),
        ("depth", len(model.blocks)),
        ("squeeze_scalar_output", model.squeeze_scalar_output),
    )


class _LatticeEquivariantBlock(StrictModule):
    first: LatticeEquivariantConvND
    normalization: TensorRMSNorm
    activation: TensorNormActivation
    second: LatticeEquivariantConvND

    def __init__(
        self,
        basis: InvariantFilterBasis,
        /,
        *,
        activation: LatticeActivation,
        use_bias: bool,
        key: Key[Array, ""],
    ):
        first_key, second_key = jr.split(key)
        self.first = LatticeEquivariantConvND(
            basis,
            use_bias=use_bias,
            key=first_key,
        )
        self.normalization = TensorRMSNorm(basis.output_layout)
        self.activation = TensorNormActivation(
            basis.output_layout,
            _activation(activation),
        )
        self.second = LatticeEquivariantConvND(
            basis,
            use_bias=False,
            key=second_key,
        )

    def __call__(
        self,
        values: Array,
        /,
        *,
        mask: Array | None,
        quadrature: Array | None,
    ) -> Array:
        branch = self.first(
            values,
            source_mask=mask,
            target_mask=mask,
            quadrature=quadrature,
        )
        branch = self.activation(self.normalization(branch))
        branch = self.second(
            branch,
            source_mask=mask,
            target_mask=mask,
            quadrature=quadrature,
        )
        output = (values + branch) / jnp.sqrt(2.0)
        if mask is not None:
            output = jnp.where(mask[..., None], output, jnp.zeros_like(output))
        return output


class LatticeEquivariantCNO(AbstractOperatorModel):
    """Periodic tensor-field operator exactly equivariant to a finite lattice group."""

    operator_architecture = "LatticeEquivariantCNO"
    _operator_contract_configuration = staticmethod(_contract_configuration)

    in_size: int
    out_size: int | Literal["scalar"]
    spatial_ndim: int
    width: int
    source_key: str | None
    squeeze_scalar_output: bool
    group: FiniteOrthogonalGroup
    input_layout: TensorFieldLayout
    hidden_layout: TensorFieldLayout
    output_layout: TensorFieldLayout
    lift: LatticeEquivariantConvND
    blocks: tuple[_LatticeEquivariantBlock, ...]
    projection: LatticeEquivariantConvND

    def __init__(
        self,
        group: FiniteOrthogonalGroup,
        input_layout: TensorFieldLayout,
        output_layout: TensorFieldLayout,
        /,
        *,
        hidden_layout: TensorFieldLayout | None = None,
        width: int = 8,
        depth: int = 4,
        kernel_size: int | Sequence[int] = 3,
        activation: LatticeActivation = "gelu",
        source_key: str | None = None,
        squeeze_scalar_output: bool = False,
        max_basis_construction_bytes: int = 256 * 1024**2,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(group, FiniteOrthogonalGroup):
            raise TypeError("group must be a FiniteOrthogonalGroup.")
        if not isinstance(input_layout, TensorFieldLayout) or not isinstance(
            output_layout, TensorFieldLayout
        ):
            raise TypeError("input_layout and output_layout must be tensor layouts.")
        if (
            input_layout.dimension != group.dimension
            or output_layout.dimension != group.dimension
        ):
            raise ValueError("Group and tensor layout dimensions must agree.")
        resolved_width = int(width)
        resolved_depth = int(depth)
        if resolved_width <= 0 or resolved_depth <= 0:
            raise ValueError("width and depth must be positive.")
        _activation(activation)
        if hidden_layout is None:
            hidden_layout = TensorFieldLayout(
                (
                    TensorFieldBlock(
                        "hidden_scalars",
                        TensorType((), dimension=group.dimension),
                        multiplicity=resolved_width,
                    ),
                )
            )
        elif not isinstance(hidden_layout, TensorFieldLayout):
            raise TypeError("hidden_layout must be a TensorFieldLayout or None.")
        if hidden_layout.dimension != group.dimension:
            raise ValueError("Hidden tensor layout and group dimensions must agree.")
        use_hidden_bias = any(
            block.tensor_type.is_scalar for block in hidden_layout.blocks
        )
        if squeeze_scalar_output and output_layout.channel_count != 1:
            raise ValueError("squeeze_scalar_output requires exactly one output channel.")

        lift_basis = InvariantFilterBasis(
            group,
            input_layout,
            hidden_layout,
            kernel_shape=kernel_size,
            max_construction_bytes=max_basis_construction_bytes,
        )
        hidden_basis = InvariantFilterBasis(
            group,
            hidden_layout,
            hidden_layout,
            kernel_shape=kernel_size,
            max_construction_bytes=max_basis_construction_bytes,
        )
        projection_basis = InvariantFilterBasis(
            group,
            hidden_layout,
            output_layout,
            kernel_shape=kernel_size,
            max_construction_bytes=max_basis_construction_bytes,
        )
        keys = jr.split(key, resolved_depth + 2)
        self.lift = LatticeEquivariantConvND(
            lift_basis,
            use_bias=use_hidden_bias,
            key=keys[0],
        )
        self.blocks = tuple(
            _LatticeEquivariantBlock(
                hidden_basis,
                activation=activation,
                use_bias=use_hidden_bias,
                key=block_key,
            )
            for block_key in keys[1:-1]
        )
        self.projection = LatticeEquivariantConvND(
            projection_basis,
            use_bias=False,
            key=keys[-1],
        )
        self.in_size = input_layout.channel_count
        self.out_size = "scalar" if squeeze_scalar_output else output_layout.channel_count
        self.spatial_ndim = group.dimension
        self.width = resolved_width
        self.source_key = source_key
        self.squeeze_scalar_output = bool(squeeze_scalar_output)
        self.group = group
        self.input_layout = input_layout
        self.hidden_layout = hidden_layout
        self.output_layout = output_layout

    def _evaluate(
        self,
        values: Array,
        axes: tuple[OperatorAxis, ...],
        /,
        *,
        source_mask: Array | None,
        target_mask: Array | None,
        source_quadrature: Array | None,
        target_quadrature: Array | None,
    ) -> Array:
        if len(axes) != self.spatial_ndim:
            raise ValueError(f"Expected {self.spatial_ndim} spatial axes.")
        if any(not axis.periodic for axis in axes):
            raise ValueError("LatticeEquivariantCNO requires periodic axes.")
        sample_shape = tuple(axis.size for axis in axes)
        if self.group.lattice_permutations is None:
            raise ValueError("The configured group has no lattice action.")
        if any(
            sample_shape[axis] != sample_shape[permutation[axis]]
            for permutation in self.group.lattice_permutations
            for axis in range(self.spatial_ndim)
        ):
            raise ValueError("Axis-permuting symmetries require equal lattice sizes.")
        hidden = self.lift(
            _prepare_grid_values(values, axes, self.in_size),
            source_mask=source_mask,
            target_mask=target_mask,
            quadrature=source_quadrature,
        )
        active_quadrature = (
            source_quadrature if target_quadrature is None else target_quadrature
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                mask=target_mask,
                quadrature=active_quadrature,
            )
        return self.projection(
            hidden,
            source_mask=target_mask,
            target_mask=target_mask,
            quadrature=active_quadrature,
        )

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        source = _operator_source(batch, self.source_key)
        query = batch.require_single_query()
        if not source.axes or not query.axes or source.values is None:
            raise ValueError(
                "LatticeEquivariantCNO requires tensor-grid source and query axes."
            )
        if source.sample_shape != query.sample_shape:
            raise ValueError(
                "LatticeEquivariantCNO requires coincident source/query grids."
            )
        source_coordinates = source.coordinates_array(case_shape=batch.case_shape)
        query_coordinates = query.coordinates_array(case_shape=batch.case_shape)
        values = eqx.error_if(
            jnp.asarray(source.values),
            jnp.any(source_coordinates != query_coordinates),
            "LatticeEquivariantCNO source/query coordinates must coincide.",
        )
        axes = source.axes
        output = self._evaluate(
            values,
            axes,
            source_mask=source.mask_array(case_shape=batch.case_shape),
            target_mask=query.mask_array(case_shape=batch.case_shape),
            source_quadrature=source.quadrature(case_shape=batch.case_shape),
            target_quadrature=query.quadrature(case_shape=batch.case_shape),
        )
        return output[..., 0] if self.squeeze_scalar_output else output

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        if isinstance(x, OperatorBatch):
            return self.__call_operator_batch__(x)
        if not isinstance(x, tuple) or len(x) != self.spatial_ndim + 1:
            raise ValueError(
                "LatticeEquivariantCNO requires (values, axis_0, ..., axis_d)."
            )
        axes = tuple(
            OperatorAxis(f"axis_{index}", nodes, periodic=True)
            for index, nodes in enumerate(x[1:])
        )
        output = self._evaluate(
            jnp.asarray(x[0]),
            axes,
            source_mask=None,
            target_mask=None,
            source_quadrature=None,
            target_quadrature=None,
        )
        return output[..., 0] if self.squeeze_scalar_output else output


__all__ = ["LatticeActivation", "LatticeEquivariantCNO"]

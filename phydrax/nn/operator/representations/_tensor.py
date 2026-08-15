#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from phydrax._strict import StrictModule


TensorVariance = Literal["contravariant", "covariant"]
TensorParity = Literal[-1, 1]


class TensorType(StrictModule):
    """Static Cartesian tensor type with index variance and parity semantics."""

    variance: tuple[TensorVariance, ...]
    parity: TensorParity
    dimension: int

    def __init__(
        self,
        variance: Sequence[TensorVariance] = (),
        /,
        *,
        parity: TensorParity = 1,
        dimension: int,
    ):
        resolved_variance = tuple(variance)
        if any(
            value not in ("contravariant", "covariant") for value in resolved_variance
        ):
            raise ValueError(
                "Tensor variance entries must be covariant or contravariant."
            )
        if parity not in (-1, 1):
            raise ValueError("Tensor parity must be +1 (ordinary) or -1 (pseudo).")
        resolved_dimension = int(dimension)
        if resolved_dimension <= 0:
            raise ValueError("Tensor dimension must be positive.")
        self.variance = resolved_variance
        self.parity = parity
        self.dimension = resolved_dimension

    @property
    def rank(self) -> int:
        return len(self.variance)

    @property
    def component_shape(self) -> tuple[int, ...]:
        return (self.dimension,) * self.rank

    @property
    def component_count(self) -> int:
        return self.dimension**self.rank

    @property
    def is_scalar(self) -> bool:
        return self.rank == 0 and self.parity == 1

    @property
    def is_pseudoscalar(self) -> bool:
        return self.rank == 0 and self.parity == -1

    def representation_matrix(self, transform: Array, /) -> Array:
        """Return the flattened component action induced by a spatial transform."""
        matrix = jnp.asarray(transform)
        expected = (self.dimension, self.dimension)
        if matrix.shape != expected:
            raise ValueError(f"transform must have shape {expected}; got {matrix.shape}.")
        if not (
            jnp.issubdtype(matrix.dtype, jnp.floating)
            or jnp.issubdtype(matrix.dtype, jnp.complexfloating)
        ):
            raise TypeError("transform must have a floating or complex floating dtype.")
        action = jnp.ones((1, 1), dtype=matrix.dtype)
        inverse_transpose = None
        for variance in self.variance:
            if variance == "contravariant":
                factor = matrix
            else:
                if inverse_transpose is None:
                    inverse_transpose = jnp.linalg.inv(matrix).T
                factor = inverse_transpose
            action = jnp.kron(action, factor)
        if self.parity == -1:
            action = jnp.linalg.det(matrix) * action
        return action

    def transform(self, values: Array, transform: Array, /) -> Array:
        """Transform unpacked values whose trailing axes are tensor components."""
        array = jnp.asarray(values)
        component_shape = self.component_shape
        if self.rank:
            if array.ndim < self.rank or array.shape[-self.rank :] != component_shape:
                raise ValueError(
                    f"Tensor values must end in component shape {component_shape}; "
                    f"got {array.shape}."
                )
            leading_shape = array.shape[: -self.rank]
        else:
            leading_shape = array.shape
        flat = array.reshape(leading_shape + (self.component_count,))
        action = self.representation_matrix(transform).astype(flat.dtype)
        transformed = oe.contract("ij,...j->...i", action, flat)
        return transformed.reshape(array.shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variance": list(self.variance),
            "parity": self.parity,
            "dimension": self.dimension,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "TensorType":
        return cls(
            value.get("variance", ()),
            parity=value.get("parity", 1),
            dimension=int(value["dimension"]),
        )


class TensorFieldBlock(StrictModule):
    """One named tensor type repeated over independent feature multiplicities."""

    name: str
    tensor_type: TensorType
    multiplicity: int

    def __init__(self, name: str, tensor_type: TensorType, /, *, multiplicity: int = 1):
        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("Tensor block names must be non-empty.")
        if not isinstance(tensor_type, TensorType):
            raise TypeError("tensor_type must be a TensorType.")
        resolved_multiplicity = int(multiplicity)
        if resolved_multiplicity <= 0:
            raise ValueError("Tensor multiplicity must be positive.")
        self.name = resolved_name
        self.tensor_type = tensor_type
        self.multiplicity = resolved_multiplicity

    @property
    def channel_count(self) -> int:
        return self.multiplicity * self.tensor_type.component_count

    @property
    def value_shape(self) -> tuple[int, ...]:
        return (self.multiplicity,) + self.tensor_type.component_shape

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tensor_type": self.tensor_type.to_dict(),
            "multiplicity": self.multiplicity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "TensorFieldBlock":
        return cls(
            str(value["name"]),
            TensorType.from_dict(value["tensor_type"]),
            multiplicity=int(value.get("multiplicity", 1)),
        )


class TensorFieldLayout(StrictModule):
    """Ordered packing contract for heterogeneous Cartesian tensor channels."""

    blocks: tuple[TensorFieldBlock, ...]

    def __init__(self, blocks: Sequence[TensorFieldBlock], /):
        resolved = tuple(blocks)
        if not resolved or any(
            not isinstance(block, TensorFieldBlock) for block in resolved
        ):
            raise TypeError(
                "blocks must be a non-empty sequence of TensorFieldBlock values."
            )
        names = tuple(block.name for block in resolved)
        if len(set(names)) != len(names):
            raise ValueError("Tensor field block names must be unique.")
        dimensions = {block.tensor_type.dimension for block in resolved}
        if len(dimensions) != 1:
            raise ValueError("All tensor field blocks must use one spatial dimension.")
        self.blocks = resolved

    @property
    def dimension(self) -> int:
        return self.blocks[0].tensor_type.dimension

    @property
    def channel_count(self) -> int:
        return sum(block.channel_count for block in self.blocks)

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks)

    def unpack(self, values: Array, /) -> tuple[Array, ...]:
        """Unpack one flat channel axis into named tensor blocks."""
        array = jnp.asarray(values)
        if array.ndim < 1 or int(array.shape[-1]) != self.channel_count:
            raise ValueError(
                f"Tensor field values must end in {self.channel_count} channels; "
                f"got {array.shape}."
            )
        leading_shape = array.shape[:-1]
        unpacked = []
        start = 0
        for block in self.blocks:
            stop = start + block.channel_count
            unpacked.append(
                array[..., start:stop].reshape(leading_shape + block.value_shape)
            )
            start = stop
        return tuple(unpacked)

    def pack(self, values: Sequence[Array] | Mapping[str, Array], /) -> Array:
        """Pack one array per tensor block into the canonical channel axis."""
        if isinstance(values, Mapping):
            if set(values) != set(self.block_names):
                raise ValueError(
                    "Tensor block mappings must match the layout names exactly."
                )
            arrays = tuple(jnp.asarray(values[block.name]) for block in self.blocks)
        else:
            arrays = tuple(jnp.asarray(value) for value in values)
            if len(arrays) != len(self.blocks):
                raise ValueError("Tensor block sequences must align with the layout.")
        leading_shape = None
        flattened = []
        for block, array in zip(self.blocks, arrays, strict=True):
            block_ndim = len(block.value_shape)
            if array.ndim < block_ndim or array.shape[-block_ndim:] != block.value_shape:
                raise ValueError(
                    f"Tensor block {block.name!r} must end in {block.value_shape}; "
                    f"got {array.shape}."
                )
            current_leading = array.shape[:-block_ndim]
            if leading_shape is None:
                leading_shape = current_leading
            elif current_leading != leading_shape:
                raise ValueError("Every tensor block must share the same leading shape.")
            flattened.append(array.reshape(current_leading + (block.channel_count,)))
        return jnp.concatenate(flattened, axis=-1)

    def transform(self, values: Array, transform: Array, /) -> Array:
        """Apply the declared tensor action to a packed channel field."""
        transformed = []
        for block, array in zip(self.blocks, self.unpack(values), strict=True):
            transformed.append(block.tensor_type.transform(array, transform))
        return self.pack(transformed)

    def channel_actions(self, transforms: Array, /) -> Array:
        """Return one packed-channel representation matrix per group element."""
        matrices = jnp.asarray(transforms)
        if matrices.ndim != 3 or matrices.shape[1:] != (self.dimension, self.dimension):
            raise ValueError(
                "transforms must have shape (group_size, dimension, dimension)."
            )
        actions = []
        for transform in matrices:
            blocks = []
            for block in self.blocks:
                component_action = block.tensor_type.representation_matrix(transform)
                blocks.append(jnp.kron(jnp.eye(block.multiplicity), component_action))
            width = self.channel_count
            packed_action = jnp.zeros((width, width), dtype=matrices.dtype)
            start = 0
            for block, action in zip(self.blocks, blocks, strict=True):
                stop = start + block.channel_count
                packed_action = packed_action.at[start:stop, start:stop].set(action)
                start = stop
            actions.append(packed_action)
        return jnp.stack(actions, axis=0)

    def validate_affine_normalization(
        self,
        scale: Sequence[float],
        offset: Sequence[float],
        /,
    ) -> None:
        """Reject channel normalization that changes the declared tensor action."""
        scales = tuple(float(value) for value in scale)
        offsets = tuple(float(value) for value in offset)
        if len(scales) != self.channel_count or len(offsets) != self.channel_count:
            raise ValueError("Tensor normalization must provide one value per channel.")
        start = 0
        for block in self.blocks:
            components = block.tensor_type.component_count
            for _ in range(block.multiplicity):
                stop = start + components
                copy_scales = scales[start:stop]
                copy_offsets = offsets[start:stop]
                if any(value != copy_scales[0] for value in copy_scales[1:]):
                    raise ValueError(
                        f"Tensor block {block.name!r} requires one scale per tensor copy."
                    )
                if not block.tensor_type.is_scalar and any(
                    value != 0.0 for value in copy_offsets
                ):
                    raise ValueError(
                        f"Non-scalar tensor block {block.name!r} requires zero offsets."
                    )
                start = stop

    def to_dict(self) -> dict[str, Any]:
        return {"blocks": [block.to_dict() for block in self.blocks]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "TensorFieldLayout":
        return cls(tuple(TensorFieldBlock.from_dict(item) for item in value["blocks"]))


__all__ = [
    "TensorFieldBlock",
    "TensorFieldLayout",
    "TensorParity",
    "TensorType",
    "TensorVariance",
]

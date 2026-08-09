#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._signature import (
    _inexact_array,
    _signature_shape,
    piecewise_linear_signature,
    PrimitiveBasis,
    tensor_logarithm,
)


def path_signature(
    values: ArrayLike,
    depth: int,
    /,
    *,
    stream: bool = False,
) -> tuple[Array, ...]:
    """Return the truncated signature of one or batched path-value arrays.

    ``values`` has shape ``batch_shape + (num_knots, dimension)``. Streaming
    output aligns with those knots and includes the empty signature at the
    initial knot.
    """
    path = _inexact_array(values)
    if path.ndim < 2 or int(path.shape[-2]) <= 0 or int(path.shape[-1]) <= 0:
        raise ValueError(
            "values must have shape batch_shape + (num_knots, dimension) "
            "with nonempty path axes."
        )
    path = eqx.error_if(
        path,
        jnp.any(~jnp.isfinite(path)),
        "Path values must be finite.",
    )
    signature = piecewise_linear_signature(
        jnp.diff(path, axis=-2),
        depth,
        stream=stream,
    )
    if not stream:
        return signature

    batch_shape = path.shape[:-2]
    sequence_axis = len(batch_shape)
    return tuple(
        jnp.concatenate(
            (
                jnp.zeros(
                    batch_shape + (1,) + level.shape[sequence_axis + 1 :],
                    dtype=level.dtype,
                ),
                level,
            ),
            axis=sequence_axis,
        )
        for level in signature
    )


def flatten_signature(
    signature: tuple[ArrayLike, ...],
    /,
    *,
    include_scalar: bool = False,
) -> Array:
    """Pack tensor levels into degree-major real feature vectors."""
    levels, dimension = _signature_shape(signature)
    batch_shape = levels[0].shape[:-1]
    flattened = tuple(
        level.reshape(batch_shape + (dimension**degree,))
        for degree, level in enumerate(levels, start=1)
    )
    if include_scalar:
        flattened = (jnp.ones(batch_shape + (1,), dtype=levels[0].dtype),) + flattened
    return jnp.concatenate(flattened, axis=-1)


def path_logsignature(
    values: ArrayLike,
    primitive_basis: PrimitiveBasis,
    /,
    *,
    stream: bool = False,
) -> Array:
    """Return standard-bracket Lyndon coordinates of a path log signature."""
    if not isinstance(primitive_basis, PrimitiveBasis):
        raise TypeError("primitive_basis must be a PrimitiveBasis.")
    signature = path_signature(values, primitive_basis.depth, stream=stream)
    return primitive_basis.tensor_to_primitive(tensor_logarithm(signature))


def repeat_last_path_padding(values: ArrayLike, lengths: ArrayLike, /) -> Array:
    """Replace every padded suffix by its final valid path value."""
    path = _inexact_array(values)
    if path.ndim < 2 or int(path.shape[-2]) <= 0 or int(path.shape[-1]) <= 0:
        raise ValueError(
            "values must have shape batch_shape + (max_knots, dimension) "
            "with nonempty path axes."
        )
    length_values = jnp.asarray(lengths)
    batch_shape = path.shape[:-2]
    if length_values.shape != batch_shape:
        raise ValueError(
            f"lengths must have shape {batch_shape}; got {length_values.shape}."
        )
    if not jnp.issubdtype(length_values.dtype, jnp.integer):
        raise TypeError("lengths must have an integer dtype.")
    max_knots = int(path.shape[-2])
    length_values = eqx.error_if(
        length_values,
        jnp.any((length_values < 1) | (length_values > max_knots)),
        "Each path length must lie between one and max_knots.",
    )
    positions = jnp.arange(max_knots, dtype=length_values.dtype)
    gather_indices = jnp.minimum(positions, length_values[..., None] - 1)
    canonical = jnp.take_along_axis(path, gather_indices[..., None], axis=-2)
    return eqx.error_if(
        canonical,
        jnp.any(~jnp.isfinite(canonical)),
        "Valid path prefixes must be finite.",
    )


def time_augment_path(
    times: ArrayLike,
    values: ArrayLike,
    /,
    *,
    lengths: ArrayLike | None = None,
) -> Array:
    """Prepend physical time, optionally canonicalizing ragged suffix padding."""
    path = _inexact_array(values)
    if path.ndim < 2 or int(path.shape[-2]) <= 0 or int(path.shape[-1]) <= 0:
        raise ValueError(
            "values must have shape batch_shape + (num_knots, dimension) "
            "with nonempty path axes."
        )
    batch_shape = path.shape[:-2]
    num_knots = int(path.shape[-2])
    time_values = _inexact_array(times)
    expected_shape = batch_shape + (num_knots,)
    if time_values.shape == (num_knots,):
        time_values = jnp.broadcast_to(time_values, expected_shape)
    elif time_values.shape != expected_shape:
        raise ValueError(
            "times must be shared or have shape "
            f"batch_shape + (num_knots,) = {expected_shape}; got {time_values.shape}."
        )
    joint = jnp.concatenate((time_values[..., None], path), axis=-1)
    if lengths is None:
        joint = eqx.error_if(
            joint,
            jnp.any(~jnp.isfinite(joint)),
            "Times and path values must be finite.",
        )
        return eqx.error_if(
            joint,
            jnp.any(jnp.diff(time_values, axis=-1) <= 0.0),
            "Times must be strictly increasing.",
        )

    length_values = jnp.asarray(lengths)
    joint = repeat_last_path_padding(joint, length_values)
    continuation = jnp.arange(max(num_knots - 1, 0)) < length_values[..., None] - 1
    return eqx.error_if(
        joint,
        jnp.any(continuation & (jnp.diff(joint[..., 0], axis=-1) <= 0.0)),
        "Valid path times must be strictly increasing.",
    )


class SignatureFeatures(StrictModule):
    """Flattened truncated tensor-signature features of complete paths."""

    dimension: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    include_scalar: bool = eqx.field(static=True)
    stream: bool = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        depth: int,
        /,
        *,
        include_scalar: bool = False,
        stream: bool = False,
    ):
        resolved_dimension = int(dimension)
        resolved_depth = int(depth)
        if resolved_dimension <= 0:
            raise ValueError("dimension must be positive.")
        if resolved_depth <= 0:
            raise ValueError("depth must be positive.")
        self.dimension = resolved_dimension
        self.depth = resolved_depth
        self.include_scalar = bool(include_scalar)
        self.stream = bool(stream)
        self.feature_id = (
            "SignatureFeatures["
            f"dimension={resolved_dimension},depth={resolved_depth},"
            f"scalar={self.include_scalar},stream={self.stream}]"
        )

    @property
    def output_size(self) -> int:
        return int(self.include_scalar) + sum(
            self.dimension**degree for degree in range(1, self.depth + 1)
        )

    def __call__(self, values: ArrayLike, /) -> Array:
        path = jnp.asarray(values)
        if path.ndim < 2 or int(path.shape[-1]) != self.dimension:
            raise ValueError(
                f"values must end in path shape (num_knots, {self.dimension})."
            )
        return flatten_signature(
            path_signature(path, self.depth, stream=self.stream),
            include_scalar=self.include_scalar,
        )


class LogSignatureFeatures(StrictModule):
    """Standard-bracket Lyndon log-signature features of complete paths."""

    primitive_basis: PrimitiveBasis
    dimension: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    stream: bool = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, depth: int, /, *, stream: bool = False):
        basis = PrimitiveBasis(dimension, depth)
        self.primitive_basis = basis
        self.dimension = basis.dimension
        self.depth = basis.depth
        self.stream = bool(stream)
        self.feature_id = (
            "LogSignatureFeatures["
            f"dimension={basis.dimension},depth={basis.depth},stream={self.stream}]"
        )

    @property
    def output_size(self) -> int:
        return self.primitive_basis.size

    def __call__(self, values: ArrayLike, /) -> Array:
        path = jnp.asarray(values)
        if path.ndim < 2 or int(path.shape[-1]) != self.dimension:
            raise ValueError(
                f"values must end in path shape (num_knots, {self.dimension})."
            )
        return path_logsignature(path, self.primitive_basis, stream=self.stream)


__all__ = [
    "LogSignatureFeatures",
    "SignatureFeatures",
    "flatten_signature",
    "path_logsignature",
    "path_signature",
    "repeat_last_path_padding",
    "time_augment_path",
]

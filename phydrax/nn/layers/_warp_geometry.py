#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ..._interpolation import apply_gather_stencil, rectilinear_stencil
from ..._strict import StrictModule
from ...linalg import inverse as matrix_inverse
from ...metrix import DENSITY_TENSOR, SCALAR_TENSOR, TensorType


WarpBoundaryMode: TypeAlias = Literal["periodic", "reflect", "clamp", "constant"]
WarpMaskMode: TypeAlias = Literal["reject", "renormalize", "strict"]

_VALID_MASK_MODES = frozenset(("reject", "renormalize", "strict"))


class RectilinearWarpDiagnostics(StrictModule):
    """Opt-in geometric diagnostics for one rectilinear warp evaluation."""

    displacement: Array
    coordinates: Array
    jacobian: Array
    determinant: Array
    interpolation_support: Array
    route_scale: Array | None

    def __init__(
        self,
        *,
        displacement: Array,
        coordinates: Array,
        jacobian: Array,
        determinant: Array,
        interpolation_support: Array,
        route_scale: Array | None = None,
    ):
        self.displacement = jnp.asarray(displacement)
        self.coordinates = jnp.asarray(coordinates)
        self.jacobian = jnp.asarray(jacobian)
        self.determinant = jnp.asarray(determinant)
        self.interpolation_support = jnp.asarray(interpolation_support, dtype=bool)
        self.route_scale = None if route_scale is None else jnp.asarray(route_scale)

    @property
    def folded_fraction(self) -> Array:
        return jnp.mean(self.determinant <= 0.0)

    @property
    def unsupported_fraction(self) -> Array:
        return 1.0 - jnp.mean(self.interpolation_support)


class GaussianWarpRoute(StrictModule):
    """Diagonal Gaussian distribution over a full displacement field."""

    mean: Array
    scale: Array

    def __init__(self, mean: Array, scale: Array, /):
        mean_ = jnp.asarray(mean)
        scale_ = jnp.asarray(scale)
        if mean_.shape != scale_.shape:
            raise ValueError("Gaussian warp mean and scale shapes must match.")
        mean_ = eqx.error_if(
            mean_,
            jnp.any(scale_ <= 0.0),
            "Gaussian warp scales must be positive.",
        )
        self.mean = mean_
        self.scale = scale_

    def sample(self, key: Array, sample_shape: Sequence[int] = (), /) -> Array:
        shape = tuple(int(size) for size in sample_shape) + self.mean.shape
        noise = jax.random.normal(key, shape, dtype=self.mean.dtype)
        return self.mean + self.scale * noise


def _canonical_axis_nodes(
    spatial_shape: tuple[int, ...],
    boundary: tuple[WarpBoundaryMode, ...],
    dtype: jnp.dtype,
    /,
) -> tuple[Array, ...]:
    nodes = []
    for size, mode in zip(spatial_shape, boundary, strict=True):
        if mode == "periodic":
            node = -1.0 + 2.0 * jnp.arange(size, dtype=dtype) / float(size)
        else:
            node = jnp.linspace(-1.0, 1.0, size, dtype=dtype)
        nodes.append(node)
    return tuple(nodes)


def normalized_axis_nodes(
    nodes: Array,
    /,
    *,
    periodic: bool,
    period: float | Array | None = None,
) -> Array:
    """Normalize ordered physical nodes to Flower's ``[-1, 1]`` convention."""

    values = jnp.asarray(nodes, dtype=float).reshape((-1,))
    if int(values.size) < 2:
        raise ValueError("A warped axis must contain at least two nodes.")
    spacing = jnp.diff(values)
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | jnp.any(spacing <= 0.0),
        "Warp axis nodes must be finite and strictly increasing.",
    )
    if periodic:
        if period is None:
            values = eqx.error_if(
                values,
                jnp.logical_not(
                    jnp.allclose(
                        spacing,
                        jnp.mean(spacing),
                        rtol=1e-5,
                        atol=1e-8,
                    )
                ),
                "A nonuniform periodic axis requires an explicit physical period.",
            )
            period_ = values[-1] - values[0] + jnp.mean(spacing)
        else:
            period_ = jnp.asarray(period, dtype=values.dtype)
        values = eqx.error_if(
            values,
            ~jnp.isfinite(period_) | (period_ <= values[-1] - values[0]),
            "Periodic axis period must exceed its sampled span.",
        )
        return -1.0 + 2.0 * (values - values[0]) / period_
    extent = values[-1] - values[0]
    return -1.0 + 2.0 * (values - values[0]) / extent


def _prepare_axis_nodes(
    axis_nodes: Sequence[Array] | None,
    spatial_shape: tuple[int, ...],
    boundary: tuple[WarpBoundaryMode, ...],
    dtype: jnp.dtype,
    /,
) -> tuple[Array, ...]:
    if axis_nodes is None:
        return _canonical_axis_nodes(spatial_shape, boundary, dtype)
    nodes = tuple(jnp.asarray(value, dtype=dtype).reshape((-1,)) for value in axis_nodes)
    if len(nodes) != len(spatial_shape):
        raise ValueError(
            f"axis_nodes must provide {len(spatial_shape)} axes; got {len(nodes)}."
        )
    for axis, (values, size, mode) in enumerate(
        zip(nodes, spatial_shape, boundary, strict=True)
    ):
        if values.shape != (size,):
            raise ValueError(
                f"axis_nodes[{axis}] must have shape {(size,)}; got {values.shape}."
            )
        spacing = jnp.diff(values)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)) | jnp.any(spacing <= 0.0),
            "Normalized warp nodes must be finite and increasing.",
        )
        outside = (values[0] < -1.0) | (
            values[-1] >= 1.0 if mode == "periodic" else values[-1] > 1.0
        )
        nodes = (
            nodes[:axis]
            + (
                eqx.error_if(
                    values,
                    outside,
                    "Normalized warp nodes must lie in the configured domain.",
                ),
            )
            + nodes[axis + 1 :]
        )
    return nodes


def _broadcast_source_mask(
    source_mask: Array,
    batch_shape: tuple[int, ...],
    spatial_shape: tuple[int, ...],
    /,
) -> Array:
    mask = jnp.asarray(source_mask, dtype=bool)
    if mask.shape == spatial_shape:
        return jnp.broadcast_to(mask, batch_shape + spatial_shape)
    expected = batch_shape + spatial_shape
    if mask.shape != expected:
        raise ValueError(
            f"source_mask must have shape {spatial_shape} or {expected}; got {mask.shape}."
        )
    return mask


def sample_rectilinear_grid(
    values: Array,
    coordinates: Array,
    /,
    *,
    spatial_ndim: int,
    boundary: Sequence[WarpBoundaryMode],
    axis_nodes: Sequence[Array] | None = None,
    source_mask: Array | None = None,
    mask_mode: WarpMaskMode = "renormalize",
    fill_value: float = 0.0,
    return_support: bool = False,
) -> Array | tuple[Array, Array]:
    """Multilinearly sample a channel-last rectilinear grid.

    Coordinates and optional ``axis_nodes`` use domain-normalized coordinates.
    Source masks are either rejected, renormalized over valid corners, or treated
    strictly so a stencil touching a hole has no support.
    """

    array = jnp.asarray(values)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    dimensions = int(spatial_ndim)
    modes = tuple(boundary)
    if dimensions not in (1, 2, 3) or len(modes) != dimensions:
        raise ValueError("Rectilinear sampling supports one, two, or three axes.")
    if mask_mode not in _VALID_MASK_MODES:
        raise ValueError("mask_mode must be 'reject', 'renormalize', or 'strict'.")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError("Rectilinear warping supports real-valued arrays only.")
    if array.ndim < dimensions + 1:
        raise ValueError("values must end in spatial dimensions and one channel axis.")
    batch_shape = tuple(int(size) for size in array.shape[: -dimensions - 1])
    spatial_shape = tuple(int(size) for size in array.shape[-dimensions - 1 : -1])
    if any(size < 2 for size in spatial_shape):
        raise ValueError("Every warped spatial axis must contain at least two nodes.")
    dtype = jnp.result_type(array.dtype, float)
    nodes = _prepare_axis_nodes(axis_nodes, spatial_shape, modes, dtype)

    query = jnp.asarray(coordinates, dtype=dtype)
    if query.ndim < len(batch_shape) + 2 or int(query.shape[-1]) != dimensions:
        raise ValueError(
            "coordinates must have shape batch_shape + query_shape + "
            f"({dimensions},); got {query.shape}."
        )
    if tuple(int(size) for size in query.shape[: len(batch_shape)]) != batch_shape:
        raise ValueError(
            f"Coordinate batch shape must be {batch_shape}; got "
            f"{query.shape[: len(batch_shape)]}."
        )
    query_shape = tuple(int(size) for size in query.shape[len(batch_shape) : -1])
    if not query_shape or any(size <= 0 for size in query_shape):
        raise ValueError("Rectilinear queries must contain at least one sample.")

    mask = None
    if source_mask is not None:
        mask = _broadcast_source_mask(source_mask, batch_shape, spatial_shape)
        if mask_mode == "reject":
            array = eqx.error_if(
                array,
                jnp.logical_not(jnp.all(mask)),
                "Rectilinear sampling in reject mode does not permit source holes.",
            )
            mask = None

    periods = tuple(2.0 if mode == "periodic" else None for mode in modes)
    stencil = rectilinear_stencil(
        nodes,
        query,
        boundary=modes,
        batch_shape=batch_shape,
        periods=periods,
        axis_bounds=((-1.0, 1.0),) * dimensions,
    )
    channels = int(array.shape[-1])
    flat_mask = None if mask is None else mask.reshape((-1,))
    interpolation = apply_gather_stencil(
        array.reshape((-1, channels)),
        stencil,
        source_mask=flat_mask,
        mask_mode=mask_mode,
    )
    support = interpolation.support
    output = jnp.where(
        support[..., None],
        interpolation.values,
        jnp.asarray(fill_value, dtype=array.dtype),
    )
    if return_support:
        return output, support
    return output


def _periodic_axis_derivative(
    values: Array,
    nodes: Array,
    axis: int,
    /,
) -> Array:
    count = int(nodes.size)
    previous_nodes = jnp.roll(nodes, 1).at[0].add(-2.0)
    next_nodes = jnp.roll(nodes, -1).at[count - 1].add(2.0)
    previous_width = nodes - previous_nodes
    next_width = next_nodes - nodes
    shape = [1] * values.ndim
    shape[axis] = count
    h0 = previous_width.reshape(shape)
    h1 = next_width.reshape(shape)
    denominator = h0 + h1
    previous_coefficient = -h1 / (h0 * denominator)
    center_coefficient = (h1 - h0) / (h0 * h1)
    next_coefficient = h0 / (h1 * denominator)
    return (
        previous_coefficient * jnp.roll(values, 1, axis=axis)
        + center_coefficient * values
        + next_coefficient * jnp.roll(values, -1, axis=axis)
    )


def warp_jacobian(
    displacement: Array,
    /,
    *,
    boundary: Sequence[WarpBoundaryMode],
    axis_nodes: Sequence[Array] | None = None,
) -> Array:
    """Return the Jacobian of ``identity + displacement`` on a rectilinear grid."""

    field = jnp.asarray(displacement)
    dimensions = int(field.shape[-1])
    modes = tuple(boundary)
    if dimensions not in (1, 2, 3) or len(modes) != dimensions:
        raise ValueError("Warp displacement must end in one, two, or three components.")
    if field.ndim < 2 * dimensions + 1 and field.ndim < dimensions + 1:
        raise ValueError("Warp displacement rank is too small for its spatial dimension.")
    spatial_shape = tuple(int(size) for size in field.shape[-dimensions - 1 : -1])
    nodes = _prepare_axis_nodes(
        axis_nodes,
        spatial_shape,
        modes,
        jnp.result_type(field.dtype, float),
    )
    case_ndim = field.ndim - dimensions - 1
    columns = []
    for local_axis, (mode, axis_values) in enumerate(zip(modes, nodes, strict=True)):
        array_axis = case_ndim + local_axis
        if mode == "periodic":
            derivative = _periodic_axis_derivative(field, axis_values, array_axis)
        else:
            derivative = jnp.asarray(jnp.gradient(field, axis_values, axis=array_axis))
        columns.append(derivative)
    gradient = jnp.stack(columns, axis=-1)
    identity = jnp.eye(dimensions, dtype=gradient.dtype)
    return gradient + identity


def normalized_lattice_from_nodes(axis_nodes: Sequence[Array], /) -> Array:
    nodes = tuple(jnp.asarray(values) for values in axis_nodes)
    if not nodes:
        raise ValueError("A normalized lattice requires at least one axis.")
    return jnp.stack(jnp.meshgrid(*nodes, indexing="ij"), axis=-1)


def _transform_tensor(
    sampled: Array,
    jacobian: Array,
    spec: TensorType,
    /,
) -> Array:
    dimensions = int(jacobian.shape[-1])
    rank = len(spec.variance)
    if rank == 0:
        transformed = sampled
    else:
        component_shape = (dimensions,) * rank
        if sampled.shape[-rank:] != component_shape:
            raise ValueError(
                f"Warped tensor must end in component shape {component_shape}; "
                f"got {sampled.shape}."
            )
        prefix = sampled.shape[:-rank]
        point_count = prod(prefix)
        transformed = sampled.reshape((point_count,) + component_shape)
        matrices = jacobian.reshape((point_count, dimensions, dimensions))
        inverse = None
        for tensor_axis, variance in enumerate(spec.variance):
            if variance == "covariant":
                matrix = jnp.swapaxes(matrices, -1, -2)
            else:
                if inverse is None:
                    inverse_result = matrix_inverse(matrices)
                    inverse = eqx.error_if(
                        inverse_result.value,
                        jnp.any(~inverse_result.successful),
                        "Warp Jacobian is singular.",
                    )
                matrix = inverse
            transformed = jnp.moveaxis(transformed, tensor_axis + 1, -1)
            transformed = oe.contract("nij,n...j->n...i", matrix, transformed)
            transformed = jnp.moveaxis(transformed, -1, tensor_axis + 1)
        transformed = transformed.reshape(sampled.shape)
    if spec.density_weight:
        determinant = jnp.abs(jnp.linalg.det(jacobian)) ** spec.density_weight
        transformed = transformed * determinant.reshape(
            determinant.shape + (1,) * len(spec.variance)
        )
    return transformed


def warp_field(
    values: Array,
    displacement: Array,
    /,
    *,
    boundary: Sequence[WarpBoundaryMode],
    axis_nodes: Sequence[Array] | None = None,
    source_mask: Array | None = None,
    mask_mode: WarpMaskMode = "renormalize",
    fill_value: float = 0.0,
    field_spec: TensorType = SCALAR_TENSOR,
    return_diagnostics: bool = False,
) -> Array | tuple[Array, RectilinearWarpDiagnostics]:
    """Transport a scalar, density, vector, covector, or tensor field by a warp."""

    field = jnp.asarray(values)
    delta = jnp.asarray(displacement)
    dimensions = int(delta.shape[-1])
    modes = tuple(boundary)
    spatial_shape = tuple(int(size) for size in delta.shape[-dimensions - 1 : -1])
    case_shape = tuple(int(size) for size in delta.shape[: -dimensions - 1])
    nodes = _prepare_axis_nodes(
        axis_nodes,
        spatial_shape,
        modes,
        jnp.result_type(field.dtype, delta.dtype, float),
    )
    tensor_shape = (dimensions,) * len(field_spec.variance)
    expected = case_shape + spatial_shape + tensor_shape
    if field.shape != expected:
        raise ValueError(f"Warped field must have shape {expected}; got {field.shape}.")
    channels = prod(tensor_shape) if tensor_shape else 1
    flattened = field.reshape(case_shape + spatial_shape + (channels,))
    lattice = normalized_lattice_from_nodes(nodes)
    coordinates = delta + lattice
    sampled_result = sample_rectilinear_grid(
        flattened,
        coordinates,
        spatial_ndim=dimensions,
        boundary=modes,
        axis_nodes=nodes,
        source_mask=source_mask,
        mask_mode=mask_mode,
        fill_value=fill_value,
        return_support=True,
    )
    sampled_flat, support = sampled_result
    sampled = sampled_flat.reshape(expected)
    jacobian = warp_jacobian(delta, boundary=modes, axis_nodes=nodes)
    transformed = _transform_tensor(sampled, jacobian, field_spec)
    if not return_diagnostics:
        return transformed
    diagnostics = RectilinearWarpDiagnostics(
        displacement=delta,
        coordinates=coordinates,
        jacobian=jacobian,
        determinant=jnp.linalg.det(jacobian),
        interpolation_support=support,
    )
    return transformed, diagnostics


def conservative_remap(
    density: Array,
    displacement: Array,
    /,
    **kwargs,
) -> Array | tuple[Array, RectilinearWarpDiagnostics]:
    """Conservatively pull back a density using the warp-map determinant."""

    return warp_field(
        density,
        displacement,
        field_spec=DENSITY_TENSOR,
        **kwargs,
    )


__all__ = [
    "GaussianWarpRoute",
    "RectilinearWarpDiagnostics",
    "WarpMaskMode",
    "conservative_remap",
    "normalized_axis_nodes",
    "normalized_lattice_from_nodes",
    "sample_rectilinear_grid",
    "warp_field",
    "warp_jacobian",
]
